# SmartPostAI 🚀

An AI-powered content intelligence and social media post generator, powered **100% by Google Gemini**. It scrapes content from target websites, summarizes articles, translates to multiple languages, and generates eye-catching social media posts with Imagen visuals.

---

## 🏗️ Architecture

- **AI Engine**: 100% **Google Gemini** via `google-genai` SDK
  - Text & Summarization: `gemini-3.6-flash`, `gemini-3.5-flash-lite`, `gemini-flash-latest`
  - Image Generation: `imagen-4.0-fast-generate-001` / `imagen-3.0-generate-002` (with Unsplash fallback)
- **Backend**: FastAPI (Python 3.10+)
  - Multi-language translation via `deep-translator`
  - Web scraping with `requests` + `BeautifulSoup4`
- **Frontend**: React 19 + Vite (built to `/static` for production)
- **Deployment**: Vercel Serverless ready via `vercel.json` and `api/index.py`

---

## ⚙️ Prerequisites & Setup

### 1. Environment Variable
Create a `.env` file in the root directory (or copy from `.env.example`):

```bash
GEMINI_API_KEY=your_gemini_api_key
```
- **Gemini API Key**: Get your key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey)

### 2. Python Virtual Environment
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Frontend Dependencies
```bash
# Install frontend packages
cd frontend
npm install
cd ..
```

---

## 💻 How to Run Locally

### Option A: Single Unified Server (Recommended)
Build the frontend once and let FastAPI serve both the backend API and the UI at `http://localhost:8000`:

```bash
# 1. Build frontend
npm run build

# 2. Start FastAPI server
source venv/bin/activate
uvicorn api.index:app --reload --port 8000
```
Open **`http://localhost:8000`** in your browser.

---

### Option B: Development Mode with Live Hot-Reloading
Run backend and frontend in separate terminals:

**Terminal 1 (Backend):**
```bash
source venv/bin/activate
uvicorn api.index:app --reload --port 8000
```

**Terminal 2 (Frontend):**
```bash
cd frontend
npm run dev
```
Open **`http://localhost:5173`** in your browser.

---

## 🚀 How to Host on Vercel

1. Push your repository to GitHub.
2. In [Vercel Dashboard](https://vercel.com/dashboard), click **Add New -> Project** and import your repo.
3. Keep default build settings (`npm run build`, output: `static`).
4. Under **Environment Variables**, add:
   - `GEMINI_API_KEY`: Your Gemini API key
5. Click **Deploy**.
