import os
import re
import json
import time
import base64
import random
import urllib.parse
from collections import Counter
from datetime import datetime
from urllib.parse import urljoin, urlparse

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup
from deep_translator import GoogleTranslator

# ── Gemini (google-genai SDK v1.x) ───────────────────────────────────────────
try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    genai = None
    genai_types = None

# ─────────────────────────────────────────────────────────────────────────────
# Model Configurations (Google Gemini)
# ─────────────────────────────────────────────────────────────────────────────
GEMINI_TEXT_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.6-flash",
    "gemini-3.1-pro-preview"
]

IMAGE_DIR = "static/generated_images"
try:
    os.makedirs(IMAGE_DIR, exist_ok=True)
except Exception:
    pass

PREFERENCE_FILE = "user_preferences.json"
PREFERENCE_HISTORY_FILE = "preference_history.json"
ARTICLES_FILE = "articles.json"
POSTS_FILE = "posts.json"

_MEM_CACHE = {
    "preferences": {},
    "history": [],
    "articles": [],
    "posts": []
}

# Standard browser headers to avoid 403 Forbidden blocks
BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Sec-Ch-Ua": '"Chromium";v="124", "Google Chrome";v="124"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"macOS"',
    "Upgrade-Insecure-Requests": "1"
}

# ─────────────────────────────────────────────────────────────────────────────
# Gemini Client
# ─────────────────────────────────────────────────────────────────────────────
def get_gemini_client():
    if genai is None:
        raise HTTPException(
            status_code=500,
            detail="google-genai package is not installed. Please run: pip install google-genai"
        )
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY environment variable is missing. Please set it in your .env file."
        )
    try:
        return genai.Client(api_key=api_key)
    except Exception as e:
        print(f"Gemini client initialization error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to initialize Gemini client: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# FastAPI App & Middleware
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="SmartPostAI API (Powered by Google Gemini)", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")
if os.path.exists("static/assets"):
    app.mount("/assets", StaticFiles(directory="static/assets"), name="assets")


class URLInput(BaseModel):
    url: str
    language: str = "en"


class UserPreferences(BaseModel):
    tone: str = "professional"
    topics: str = "general"
    language: str = "en"


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


# ── Storage Helpers (Stored in /tmp so project directory stays clean) ────────

def _safe_write_json(filename: str, data):
    target = os.path.join("/tmp", os.path.basename(filename))
    try:
        with open(target, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    except Exception:
        pass

def _safe_read_json(filename: str, default=None):
    target = os.path.join("/tmp", os.path.basename(filename))
    if os.path.exists(target):
        try:
            with open(target, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return default if default is not None else {}


# ── Single-Pass Scraping Helper ──────────────────────────────────────────────

def is_same_domain(url1: str, url2: str) -> bool:
    try:
        netloc1 = urlparse(url1).netloc.replace("www.", "")
        netloc2 = urlparse(url2).netloc.replace("www.", "")
        return netloc1 == netloc2
    except Exception:
        return False


def slug_to_title(url: str) -> str:
    try:
        path = urlparse(url).path.strip("/").split("/")[-1]
        if path:
            words = re.split(r"[-_.]", path)
            cleaned = " ".join(w.capitalize() for w in words if w and not w.isdigit())
            if cleaned:
                return cleaned
        domain = urlparse(url).netloc.replace("www.", "").split(".")[0]
        return domain.capitalize() if domain else "Featured Article"
    except Exception:
        return "Featured Article"


def scrape_page(url: str) -> dict:
    result = {
        "url": url,
        "title": slug_to_title(url),
        "description": "",
        "content": "",
        "links": []
    }

    try:
        resp = requests.get(url, headers=BROWSER_HEADERS, timeout=6)
        if resp.status_code == 200 and resp.text:
            soup = BeautifulSoup(resp.text, "html.parser")

            og_title = soup.find("meta", property="og:title")
            title_tag = soup.find("title")
            h1_tag = soup.find("h1")

            if og_title and og_title.get("content"):
                result["title"] = og_title.get("content").strip()
            elif title_tag and title_tag.string:
                result["title"] = title_tag.string.strip()
            elif h1_tag:
                result["title"] = h1_tag.get_text().strip()

            meta_desc = soup.find("meta", attrs={"name": "description"}) or soup.find("meta", property="og:description")
            if meta_desc and meta_desc.get("content"):
                result["description"] = meta_desc.get("content").strip()

            main = (
                soup.find("main") or
                soup.find("article") or
                soup.find("div", class_=re.compile(r"content|body|post|article", re.I)) or
                soup.find("body")
            )
            if main:
                for s in main(["script", "style", "nav", "footer", "header"]):
                    s.decompose()
                text = main.get_text(separator=" ", strip=True)
                result["content"] = " ".join(text.split()[:500])

            seen = {url}
            for tag in soup.find_all("a", href=True):
                full_url = urljoin(url, tag["href"]).split("#")[0].split("?")[0]
                if is_same_domain(full_url, url) and full_url not in seen and len(full_url) > len(url) - 5:
                    seen.add(full_url)
                    result["links"].append(full_url)
    except Exception as e:
        print(f"scrape_page note for {url}: {e}")

    if not result["content"]:
        result["content"] = f"Topic related to {result['title']} from {urlparse(url).netloc}."

    return result


# ── Preferences ───────────────────────────────────────────────────────────────

def save_user_preferences(preferences: dict):
    now = datetime.now().isoformat()
    existing = _safe_read_json(PREFERENCE_FILE, {})
    if not isinstance(existing, dict):
        existing = {}
    existing.update(preferences)
    _MEM_CACHE["preferences"] = existing
    _safe_write_json(PREFERENCE_FILE, existing)

    history = _safe_read_json(PREFERENCE_HISTORY_FILE, [])
    if not isinstance(history, list):
        history = []
    history.append({"timestamp": now, "preferences": existing})
    _MEM_CACHE["history"] = history
    _safe_write_json(PREFERENCE_HISTORY_FILE, history)


def load_user_preferences() -> dict:
    if _MEM_CACHE["preferences"]:
        return _MEM_CACHE["preferences"]
    data = _safe_read_json(PREFERENCE_FILE, {})
    if isinstance(data, dict):
        _MEM_CACHE["preferences"] = data
        return data
    return {}


def analyze_preferences() -> str:
    history = _MEM_CACHE["history"] or _safe_read_json(PREFERENCE_HISTORY_FILE, [])
    if not history or not isinstance(history, list):
        return "No preference history available."

    tone_counter = Counter(e.get("preferences", {}).get("tone", "Not specified") for e in history if isinstance(e, dict))
    most_common_tone = tone_counter.most_common(1)[0][0] if tone_counter else "Not specified"

    all_topics = []
    for e in history:
        if isinstance(e, dict):
            topics_str = e.get("preferences", {}).get("topics", "")
            all_topics.extend(t.strip() for t in topics_str.split(",") if t.strip())
    topic_counter = Counter(all_topics)

    out = f"Most common tone: {most_common_tone}\n"
    out += "Top 5 topics of interest:\n"
    if topic_counter:
        for topic, count in topic_counter.most_common(5):
            out += f"  - {topic}: {count} occurrences\n"
    else:
        out += "  - No topics recorded yet\n"
    return out


# ── AI Generation (100% Google Gemini) ───────────────────────────────────────

def generate_social_content_gemini(page_data: dict, preferences: dict, language: str) -> dict:
    tone = preferences.get("tone", "engaging and professional")
    topics = preferences.get("topics", "general")
    url = page_data.get("url", "")
    title = page_data.get("title", "")
    content = page_data.get("content", "")

    lang_names = {
        "en": "English",
        "hi": "Hindi",
        "fr": "French",
        "de": "German",
        "es": "Spanish",
        "ja": "Japanese",
        "zh-CN": "Chinese",
        "ar": "Arabic"
    }
    target_lang = lang_names.get(language, "English")

    prompt = f"""You are a world-class AI content creator.
Analyze the following webpage/topic and generate engaging social media assets in {target_lang}.

URL: {url}
Title: {title}
Context/Content: {content[:2000]}
Tone: {tone}
Topics of Interest: {topics}

Return a valid JSON object ONLY with the following structure (no markdown fences, just pure JSON):
{{
  "summary": "A clear, compelling 2-3 sentence summary of the article or brand in {target_lang}.",
  "post_content": "A high-impact social media post (max 220 chars) in {target_lang} with relevant hashtags and a strong call to action.\\n\\n{url}",
  "image_prompt": "A 1-sentence prompt describing a clean, photorealistic visual for this topic (no text, no overlays)."
}}
"""
    client = get_gemini_client()

    for model_name in GEMINI_TEXT_MODELS:
        try:
            resp = client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            if resp and resp.text:
                raw = resp.text.strip()
                clean_json = re.sub(r"^```json\s*", "", raw)
                clean_json = re.sub(r"^```\s*", "", clean_json)
                clean_json = re.sub(r"\s*```$", "", clean_json).strip()
                data = json.loads(clean_json)
                return data
        except Exception as e:
            print(f"Gemini model '{model_name}' prompt note: {e}")
            continue

    # Fallback if AI JSON parse failed
    return {
        "summary": f"{title} - Discover the latest updates and insights.",
        "post_content": f"Check out {title}! Exploring the key highlights and updates.\n\n{url}",
        "image_prompt": f"Professional photography of {title}"
    }


def generate_visual_image(image_prompt: str, title: str, idx: int) -> str:
    """
    Generates a reliable, photorealistic visual image.
    Attempts to fetch and embed as Base64 for 100% offline & cross-origin display guarantee.
    """
    clean_p = re.sub(r"[^a-zA-Z0-9\s,.-]", "", image_prompt).strip()
    if not clean_p:
        clean_p = f"{title} photography"

    encoded = urllib.parse.quote(clean_p[:120])
    ai_url = f"https://image.pollinations.ai/prompt/{encoded}?width=800&height=500&nologo=true&seed={random.randint(100, 99999)}"

    # Try downloading and converting to base64 for instant display in any browser
    try:
        resp = requests.get(ai_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=6)
        if resp.status_code == 200 and len(resp.content) > 1000:
            b64 = base64.b64encode(resp.content).decode("utf-8")
            return f"data:image/jpeg;base64,{b64}"
    except Exception:
        pass

    # Return direct CDN URL if timeout occurred
    return ai_url


# ── API Routes ────────────────────────────────────────────────────────────────

@app.get("/")
async def read_index():
    if os.path.exists("static/index.html"):
        return FileResponse("static/index.html")
    return {"message": "SmartPostAI API is running. Build frontend with 'npm run build' to serve the UI."}


@app.get("/favicon.svg")
async def get_favicon():
    if os.path.exists("static/favicon.svg"):
        return FileResponse("static/favicon.svg")
    return JSONResponse(status_code=404, content={"detail": "Not found"})


@app.get("/icons.svg")
async def get_icons():
    if os.path.exists("static/icons.svg"):
        return FileResponse("static/icons.svg")
    return JSONResponse(status_code=404, content={"detail": "Not found"})


@app.post("/save_preferences")
async def save_preferences(preferences: UserPreferences):
    save_user_preferences(preferences.model_dump() if hasattr(preferences, "model_dump") else preferences.dict())
    return {"message": "Preferences saved successfully"}


@app.get("/analyze_preferences")
async def get_preference_analysis():
    return {"analysis": analyze_preferences()}


@app.post("/scrape_and_generate")
async def scrape_and_generate(url_input: URLInput):
    base_url = url_input.url.strip()
    language = url_input.language or "en"
    prefs = load_user_preferences()

    # 1. Scrape base page in a single pass
    root_page = scrape_page(base_url)

    pages_to_process = [root_page]
    for link in root_page["links"][:2]:
        pages_to_process.append(scrape_page(link))

    articles = []
    posts = []
    all_content = ""

    for idx, page in enumerate(pages_to_process):
        all_content += page["content"] + " "

        # Generate structured content in a single Gemini call
        ai_data = generate_social_content_gemini(page, prefs, language)

        summary = ai_data.get("summary", "")
        post_content = ai_data.get("post_content", "")
        img_prompt = ai_data.get("image_prompt", page["title"])

        article = {
            "title": page["title"],
            "description": summary,
            "url": page["url"]
        }
        articles.append(article)

        # Generate real visual image
        image_url = generate_visual_image(img_prompt, page["title"], idx)

        posts.append({
            "post_content": post_content,
            "url": page["url"],
            "image_url": image_url
        })

    # Cache results
    _MEM_CACHE["articles"] = articles
    _MEM_CACHE["posts"] = posts

    _safe_write_json(ARTICLES_FILE, articles)
    _safe_write_json(POSTS_FILE, posts)

    return {
        "message": f"Done! {len(articles)} articles summarised, {len(posts)} social posts created.",
        "articles": articles,
        "posts": posts
    }


@app.get("/posts.json")
async def read_posts():
    data = _MEM_CACHE["posts"] or _safe_read_json(POSTS_FILE, [])
    return JSONResponse(content=data)


@app.get("/articles.json")
async def read_articles():
    data = _MEM_CACHE["articles"] or _safe_read_json(ARTICLES_FILE, [])
    return JSONResponse(content=data)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)