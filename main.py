from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import nltk
import json
from openai import OpenAI
from urllib.parse import urljoin, urlparse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import random
import os
from collections import Counter
from datetime import datetime
from deep_translator import GoogleTranslator


nltk.download('stopwords')

app = FastAPI()

class URLInput(BaseModel):
    url: str
    language: str

class UserPreferences(BaseModel):
    tone: str
    topics: str
    language: str

# Initialize the OpenAI client
client = OpenAI(api_key="API_KEY")  # Replace with your actual OpenAI API key



app.mount("/static", StaticFiles(directory="static"), name="static")

PREFERENCE_FILE = "user_preferences.json"
PREFERENCE_HISTORY_FILE = "preference_history.json"

@app.get("/")
async def read_index():
    return FileResponse('static/index.html')

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={"detail": str(exc)}
    )

def is_same_domain(url1, url2):
    return urlparse(url1).netloc == urlparse(url2).netloc

def scrape_urls(base_url):
    response = requests.get(base_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    return [urljoin(base_url, link.get('href')) for link in soup.find_all('a') if link.get('href') and is_same_domain(urljoin(base_url, link.get('href')), base_url)]

def scrape_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content')
    return main_content.get_text() if main_content else soup.get_text()

def clean_content(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.lower().split()
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    return ' '.join(list(dict.fromkeys(words)))

def save_user_preferences(preferences):
    current_time = datetime.now().isoformat()
    
    if os.path.exists(PREFERENCE_FILE):
        with open(PREFERENCE_FILE, 'r', encoding="utf-8") as file:
            existing_preferences = json.load(file)
    else:
        existing_preferences = {}

    existing_preferences.update(preferences)

    with open(PREFERENCE_FILE, 'w', encoding="utf-8") as file:
        json.dump(existing_preferences, file, ensure_ascii=False, indent=4)

    if os.path.exists(PREFERENCE_HISTORY_FILE):
        with open(PREFERENCE_HISTORY_FILE, 'r', encoding="utf-8") as file:
            preference_history = json.load(file)
    else:
        preference_history = []

    preference_history.append({
        "timestamp": current_time,
        "preferences": existing_preferences
    })

    with open(PREFERENCE_HISTORY_FILE, 'w', encoding="utf-8") as file:
        json.dump(preference_history, file, ensure_ascii=False, indent=4)

def load_user_preferences():
    if os.path.exists(PREFERENCE_FILE):
        with open(PREFERENCE_FILE, 'r', encoding="utf-8") as file:
            return json.load(file)
    return {}

def analyze_preferences():
    if not os.path.exists(PREFERENCE_HISTORY_FILE):
        return "No preference history available."

    with open(PREFERENCE_HISTORY_FILE, 'r', encoding="utf-8") as file:
        preference_history = json.load(file)

    if not preference_history:
        return "Preference history is empty."

    tone_counter = Counter()
    for entry in preference_history:
        tone_counter[entry['preferences'].get('tone', 'Not specified')] += 1

    most_common_tone = tone_counter.most_common(1)[0][0]

    all_topics = []
    for entry in preference_history:
        topics = entry['preferences'].get('topics', '').split(',')
        all_topics.extend([topic.strip() for topic in topics if topic.strip()])

    topic_counter = Counter(all_topics)
    most_common_topics = topic_counter.most_common(5)

    analysis = f"Most common tone: {most_common_tone}\n"
    analysis += "Top 5 topics of interest:\n"
    for topic, count in most_common_topics:
        analysis += f"- {topic}: {count} occurrences\n"

    return analysis



def translate_text(text, target_language):
    if not text or target_language == 'en':
        return text

    try:
        # Split long text into chunks to avoid length limitations
        max_chunk_size = 4500  # Google Translate has a character limit
        chunks = [text[i:i + max_chunk_size] for i in range(0, len(text), max_chunk_size)]
        
        translated_chunks = []
        translator = GoogleTranslator(source='auto', target=target_language)
        
        for chunk in chunks:
            translated_chunk = translator.translate(chunk)
            if translated_chunk:
                translated_chunks.append(translated_chunk)
            else:
                translated_chunks.append(chunk)
        
        return ' '.join(translated_chunks)
    except Exception as e:
        print(f"Translation error: {str(e)}")
        return text

def generate_content_with_preferences(prompt, preferences):
    enhanced_prompt = f"Consider these user preferences: Tone: {preferences.get('tone', 'Not specified')}, " \
                      f"Topics: {preferences.get('topics', 'Not specified')}. " \
                      f"Now, based on these preferences, {prompt}"

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an AI assistant that generates content based on user preferences."},
            {"role": "user", "content": enhanced_prompt}
        ],
        temperature=0.7,
        max_tokens=300 
    )

    content = response.choices[0].message.content.strip()
    
    '''if preferences.get('language', 'en') != 'en':
        content = translate_text(content, preferences['language'])'''

    return content

def generate_article(content, url, language):
    preferences = load_user_preferences()
    preferences['language'] = language
    prompt = f"Generate a short summary of the article from this URL: {url}\n\nContent: {content[:1500]}"
    
    # Generate content in English first
    english_content = generate_content_with_preferences(prompt, preferences)
    
    # Translate if needed
    if language != 'en':
        return translate_text(english_content, language)
    return english_content

def extract_title(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    title = soup.find('title')
    if title:
        return title.string
    h1 = soup.find('h1')
    if h1:
        return h1.text
    return "Untitled Article"

def generate_image(prompt):
    try:
        response = client.images.generate(
            prompt=prompt,
            size="1024x1024",
            n=1,
        )
        return response.data[0].url
    except Exception as e:
        print(f"Error generating image: {str(e)}")
        return None

def generate_social_media_post(article, language):
    preferences = load_user_preferences()
    preferences['language'] = language

    post_prompt = f"Create an engaging and catchy social media post (maximum 280 characters) for the following article. Include the article URL at the end of the post.\n\nDescription: {article['description']}\nURL: {article['url']}"

    post_content = generate_content_with_preferences(post_prompt, preferences)
    
    # Translate post content if needed
    if language != 'en':
        post_content = translate_text(post_content, language)
    
    image_prompt = f"Generate a short, descriptive prompt for an image that would go well with this social media post and take care of spelling mistakes. : {post_content}"
    image_prompt_content = generate_content_with_preferences(image_prompt, preferences)
    image_url = generate_image(image_prompt_content)
    
    return {
        "post_content": post_content,
        "url": article['url'],
        "image_url": image_url
    }

@app.post("/save_preferences")
async def save_preferences(preferences: UserPreferences):
    save_user_preferences(preferences.dict())
    return {"message": "Preferences saved successfully"}

@app.get("/analyze_preferences")
async def get_preference_analysis():
    analysis = analyze_preferences()
    return {"analysis": analysis}

@app.post("/scrape_and_generate")
async def scrape_and_generate(url_input: URLInput):
    base_url = url_input.url
    language = url_input.language
    urls = scrape_urls(base_url)

    articles = []
    posts = []
    all_content = ""

    if len(urls) > 10:
        start_index = len(urls) // 4
        end_index = 3 * len(urls) // 4
        middle_urls = urls[start_index:end_index]
        selected_urls = random.sample(middle_urls, min(5, len(middle_urls)))
    elif len(urls) > 5:
        selected_urls = random.sample(urls[1:-1], min(5, len(urls) - 2))
    else:
        selected_urls = urls

    for url in selected_urls:
        content = scrape_content(url)
        cleaned_content = clean_content(content)
        all_content += cleaned_content + " "

        article_summary = generate_article(content, url, language)
        title = extract_title(url)
        '''if language != 'en':
            title = translate_text(title, language)'''

        article = {
            "title": title,
            "description": article_summary,
            "url": url
        }

        articles.append(article)

        social_media_post = generate_social_media_post(article, language)
        posts.append(social_media_post)

    with open("cleaned_content.txt", "w", encoding="utf-8") as f:
        f.write(all_content)

    with open("articles.json", "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=4)

    with open("posts.json", "w", encoding="utf-8") as f:
        json.dump(posts, f, ensure_ascii=False, indent=4)

    return {"message": "Scraping, article generation, social media post creation, and image generation completed successfully. Check articles.json and posts.json for results."}

@app.get("/posts.json")
async def read_posts():
    return FileResponse('posts.json')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)