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
from urllib.parse import urljoin, urlparse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import random
import os
import base64
from dotenv import load_dotenv

# ── Groq (text) ──────────────────────────────────────────────────────────────
from groq import Groq

# ── Gemini (images) — google-genai SDK v1.x ──────────────────────────────────
from google import genai
from google.genai import types as genai_types

load_dotenv()
from collections import Counter
from datetime import datetime
from deep_translator import GoogleTranslator

nltk.download('stopwords')

# ─────────────────────────────────────────────────────────────────────────────
# Clients
#   GROQ_API_KEY   → https://console.groq.com           (free)
#   GEMINI_API_KEY → https://aistudio.google.com/apikey (free, ~500 imgs/day)
# ─────────────────────────────────────────────────────────────────────────────
groq_client      = Groq(api_key=os.getenv("GROQ_API_KEY"))
GROQ_MODEL       = "llama-3.3-70b-versatile"        # best free Groq model

gemini_client    = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
GEMINI_IMG_MODEL = "imagen-4.0-fast-generate-001"            # natively supported image model

# Folder to save generated images so FastAPI can serve them
IMAGE_DIR = "static/generated_images"
os.makedirs(IMAGE_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI()


class URLInput(BaseModel):
    url: str
    language: str


class UserPreferences(BaseModel):
    tone: str
    topics: str
    language: str


app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/assets", StaticFiles(directory="static/assets"), name="assets")

PREFERENCE_FILE         = "user_preferences.json"
PREFERENCE_HISTORY_FILE = "preference_history.json"


@app.get("/")
async def read_index():
    return FileResponse("static/index.html")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


# ── Scraping helpers ──────────────────────────────────────────────────────────

def is_same_domain(url1, url2):
    return urlparse(url1).netloc == urlparse(url2).netloc


def scrape_urls(base_url: str) -> list:
    try:
        resp = requests.get(base_url, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        seen, links = set(), []
        for tag in soup.find_all("a"):
            href = tag.get("href")
            if not href:
                continue
            full = urljoin(base_url, href)
            if is_same_domain(full, base_url) and full not in seen:
                seen.add(full)
                links.append(full)
        return links
    except Exception as e:
        print(f"scrape_urls error: {e}")
        return []


def scrape_content(url: str) -> str:
    try:
        resp = requests.get(url, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        main = (
            soup.find("main") or
            soup.find("article") or
            soup.find("div", class_="content")
        )
        return main.get_text() if main else soup.get_text()
    except Exception as e:
        print(f"scrape_content error {url}: {e}")
        return ""


def clean_content(text: str) -> str:
    text  = re.sub(r"[^a-zA-Z\s]", "", text)
    words = text.lower().split()
    stop  = set(stopwords.words("english"))
    return " ".join(list(dict.fromkeys(w for w in words if w not in stop)))


def extract_title(url: str) -> str:
    try:
        resp = requests.get(url, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        t = soup.find("title")
        if t and t.string:
            return t.string.strip()
        h = soup.find("h1")
        if h:
            return h.get_text().strip()
    except Exception as e:
        print(f"extract_title error: {e}")
    return "Untitled Article"


# ── Preferences ───────────────────────────────────────────────────────────────

def save_user_preferences(preferences: dict):
    now = datetime.now().isoformat()
    existing = {}
    if os.path.exists(PREFERENCE_FILE):
        with open(PREFERENCE_FILE, "r", encoding="utf-8") as f:
            existing = json.load(f)
    existing.update(preferences)
    with open(PREFERENCE_FILE, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=4)

    history = []
    if os.path.exists(PREFERENCE_HISTORY_FILE):
        with open(PREFERENCE_HISTORY_FILE, "r", encoding="utf-8") as f:
            history = json.load(f)
    history.append({"timestamp": now, "preferences": existing})
    with open(PREFERENCE_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=4)


def load_user_preferences() -> dict:
    if os.path.exists(PREFERENCE_FILE):
        with open(PREFERENCE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def analyze_preferences() -> str:
    if not os.path.exists(PREFERENCE_HISTORY_FILE):
        return "No preference history available."
    with open(PREFERENCE_HISTORY_FILE, "r", encoding="utf-8") as f:
        history = json.load(f)
    if not history:
        return "Preference history is empty."

    tone_counter = Counter(e["preferences"].get("tone", "Not specified") for e in history)
    most_common_tone = tone_counter.most_common(1)[0][0]

    all_topics = []
    for e in history:
        all_topics.extend(
            t.strip() for t in e["preferences"].get("topics", "").split(",") if t.strip()
        )
    topic_counter = Counter(all_topics)

    out  = f"Most common tone: {most_common_tone}\n"
    out += "Top 5 topics of interest:\n"
    for topic, count in topic_counter.most_common(5):
        out += f"  - {topic}: {count} occurrences\n"
    return out


# ── Translation ───────────────────────────────────────────────────────────────

def translate_text(text: str, target_language: str) -> str:
    if not text or target_language == "en":
        return text
    try:
        translator = GoogleTranslator(source="auto", target=target_language)
        chunks = [text[i:i + 4500] for i in range(0, len(text), 4500)]
        return " ".join(translator.translate(c) or c for c in chunks)
    except Exception as e:
        print(f"translate_text error: {e}")
        return text


# ── Text generation — Groq (free, llama-3.3-70b) ─────────────────────────────

def generate_text(prompt: str, preferences: dict) -> str:
    """
    Uses Groq's free tier — llama-3.3-70b-versatile.
    Raises HTTP 503 on failure so the frontend gets a clean error message.
    """
    tone   = preferences.get("tone", "professional")
    topics = preferences.get("topics", "general")

    system = (
        f"You are a smart, concise content assistant. "
        f"Always write in a {tone} tone. "
        f"The reader is interested in: {topics}."
    )
    try:
        resp = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.7,
            max_tokens=400,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"Groq error: {e}")
        raise HTTPException(status_code=503, detail=f"Text generation failed: {e}")


# ── Image generation — Gemini ──────────────────────────────────────────────────

def generate_image_gemini(image_prompt: str, filename: str) -> str | None:
    """
    Native image generation using Imagen 4 via the google-genai SDK.
    Returns a public URL path (e.g., /static/generated_images/post_xxx.png) on success.
    """
    try:
        response = gemini_client.models.generate_images(
            model=GEMINI_IMG_MODEL,
            prompt=(
                f"Generate a visually striking, photorealistic image suitable for a "
                f"social media post. No text overlays. High quality. "
                f"Theme: {image_prompt}"
            ),
            config=genai_types.GenerateImagesConfig(
                number_of_images=1,
                output_mime_type="image/png"
            ),
        )

        if response.generated_images:
            img_bytes = response.generated_images[0].image.image_bytes
            filepath = os.path.join(IMAGE_DIR, filename)
            with open(filepath, "wb") as fh:
                fh.write(img_bytes)
            return f"/static/generated_images/{filename}"

        print("Gemini: no image part in response.")
        return None

    except Exception as e:
        print(f"Gemini image error: {e}")
        return None


# ── Article & social post generation ─────────────────────────────────────────

def generate_article(content: str, url: str, language: str) -> str:
    prefs = load_user_preferences()
    prefs["language"] = language

    prompt = (
        f"Write a clear, engaging 3-4 sentence summary of the article below.\n"
        f"URL: {url}\n\n"
        f"Content (first 2000 chars):\n{content[:2000]}"
    )
    summary = generate_text(prompt, prefs)
    return translate_text(summary, language) if language != "en" else summary


def generate_social_media_post(article: dict, language: str, post_index: int) -> dict:
    prefs = load_user_preferences()
    prefs["language"] = language

    # 1. Write the post copy
    post_prompt = (
        f"Write a punchy social media post (max 240 characters, URL not included). "
        f"Be engaging, relevant, and end with a call to action. "
        f"Then on a new line add: {article['url']}\n\n"
        f"Title: {article['title']}\n"
        f"Summary: {article['description']}"
    )
    post_content = generate_text(post_prompt, prefs)
    if language != "en":
        post_content = translate_text(post_content, language)

    # 2. Ask Groq to write a short image prompt
    img_prompt_request = (
        f"Write a 1-sentence image generation prompt for a social media post about: "
        f"'{article['title']}'. Focus on one powerful visual. No text in the image."
    )
    img_prompt = generate_text(img_prompt_request, {"tone": "neutral", "topics": "visual"})

    # 3. Generate image with Gemini (free tier)
    image_filename = f"post_{post_index}_{random.randint(1000, 9999)}.png"
    image_url = generate_image_gemini(img_prompt, image_filename)

    # 4. Graceful fallback to Unsplash if Gemini fails
    if not image_url:
        kw = "+".join(article["title"].split()[:4])
        image_url = f"https://source.unsplash.com/1024x512/?{kw}"

    return {
        "post_content": post_content,
        "url":          article["url"],
        "image_url":    image_url,
    }


# ── API routes ────────────────────────────────────────────────────────────────

@app.post("/save_preferences")
async def save_preferences(preferences: UserPreferences):
    save_user_preferences(preferences.dict())
    return {"message": "Preferences saved successfully"}


@app.get("/analyze_preferences")
async def get_preference_analysis():
    return {"analysis": analyze_preferences()}


@app.post("/scrape_and_generate")
async def scrape_and_generate(url_input: URLInput):
    base_url = url_input.url
    language = url_input.language

    urls = scrape_urls(base_url)
    if not urls:
        raise HTTPException(status_code=400, detail="No URLs found on the provided page.")

    # Pick a representative middle sample (avoids nav/footer junk at the edges)
    if len(urls) > 10:
        s, e     = len(urls) // 4, 3 * len(urls) // 4
        pool     = urls[s:e]
        selected = random.sample(pool, min(5, len(pool)))
    elif len(urls) > 5:
        selected = random.sample(urls[1:-1], min(5, len(urls) - 2))
    else:
        selected = urls

    articles    = []
    posts       = []
    all_content = ""

    for idx, url in enumerate(selected):
        content      = scrape_content(url)
        all_content += clean_content(content) + " "

        summary = generate_article(content, url, language)
        title   = extract_title(url)

        article = {"title": title, "description": summary, "url": url}
        articles.append(article)
        posts.append(generate_social_media_post(article, language, idx))

    with open("cleaned_content.txt", "w", encoding="utf-8") as f:
        f.write(all_content)
    with open("articles.json", "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=4)
    with open("posts.json", "w", encoding="utf-8") as f:
        json.dump(posts, f, ensure_ascii=False, indent=4)

    return {
        "message": (
            f"Done! {len(articles)} articles summarised, "
            f"{len(posts)} social posts created with AI-generated images."
        )
    }


@app.get("/posts.json")
async def read_posts():
    return FileResponse("posts.json")


@app.get("/articles.json")
async def read_articles():
    return FileResponse("articles.json")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)