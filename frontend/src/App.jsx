import { useState, useEffect, useRef } from "react";

const styles = `
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:ital,wght@0,400;0,500;1,400&family=Instrument+Serif:ital@0;1&display=swap');

  * { margin: 0; padding: 0; box-sizing: border-box; }

  :root {
    --bg: #0a0a0f;
    --surface: #111118;
    --surface2: #1a1a26;
    --border: rgba(255,255,255,0.07);
    --accent: #7c6aff;
    --accent2: #ff6a8e;
    --accent3: #6affce;
    --text: #f0f0f8;
    --muted: #6b6b80;
    --card-glow: rgba(124,106,255,0.08);
  }

  body { background: var(--bg); color: var(--text); font-family: 'DM Mono', monospace; }

  .app {
    min-height: 100vh;
    background: var(--bg);
    position: relative;
    overflow-x: hidden;
  }

  /* Noise texture overlay */
  .app::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.04'/%3E%3C/svg%3E");
    pointer-events: none;
    z-index: 0;
    opacity: 0.4;
  }

  /* Gradient mesh bg */
  .mesh {
    position: fixed;
    width: 800px; height: 800px;
    border-radius: 50%;
    filter: blur(120px);
    pointer-events: none;
    z-index: 0;
  }
  .mesh-1 { background: radial-gradient(circle, rgba(124,106,255,0.12), transparent 70%); top: -200px; left: -200px; }
  .mesh-2 { background: radial-gradient(circle, rgba(255,106,142,0.08), transparent 70%); bottom: -200px; right: -200px; }
  .mesh-3 { background: radial-gradient(circle, rgba(106,255,206,0.06), transparent 70%); top: 40%; left: 40%; }

  /* Header */
  .header {
    position: relative; z-index: 10;
    padding: 2rem 3rem;
    display: flex; align-items: center; justify-content: space-between;
    border-bottom: 1px solid var(--border);
    backdrop-filter: blur(20px);
    background: rgba(10,10,15,0.6);
    position: sticky; top: 0;
  }

  .logo {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 1.4rem;
    letter-spacing: -0.03em;
    display: flex; align-items: center; gap: 0.6rem;
  }
  .logo-dot { width: 8px; height: 8px; background: var(--accent); border-radius: 50%; box-shadow: 0 0 12px var(--accent); animation: pulse 2s ease-in-out infinite; }
  @keyframes pulse { 0%,100% { opacity:1; transform:scale(1); } 50% { opacity:0.6; transform:scale(0.8); } }

  .nav-tabs {
    display: flex; gap: 0.25rem;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 4px;
  }
  .nav-tab {
    padding: 0.5rem 1.2rem;
    border-radius: 8px;
    border: none;
    background: transparent;
    color: var(--muted);
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    cursor: pointer;
    transition: all 0.2s;
    letter-spacing: 0.05em;
  }
  .nav-tab:hover { color: var(--text); }
  .nav-tab.active { background: var(--surface2); color: var(--text); border: 1px solid var(--border); }

  .badge {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    padding: 4px 10px;
    border-radius: 20px;
    border: 1px solid rgba(124,106,255,0.3);
    color: var(--accent);
    letter-spacing: 0.1em;
  }

  /* Main layout */
  .main { position: relative; z-index: 1; padding: 3rem; max-width: 1200px; margin: 0 auto; }

  /* Hero section */
  .hero { margin-bottom: 3rem; }
  .hero-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: var(--accent3);
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 1rem;
    display: flex; align-items: center; gap: 0.5rem;
  }
  .hero-eyebrow::before { content: ''; display: block; width: 24px; height: 1px; background: var(--accent3); }
  .hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.5rem, 5vw, 4rem);
    font-weight: 800;
    line-height: 1.05;
    letter-spacing: -0.04em;
    margin-bottom: 1rem;
  }
  .hero-title span { font-family: 'Instrument Serif', serif; font-style: italic; color: var(--accent); }
  .hero-sub { color: var(--muted); font-size: 0.85rem; max-width: 500px; line-height: 1.7; }

  /* Cards */
  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem;
    position: relative;
    transition: border-color 0.3s, transform 0.3s;
  }
  .card:hover { border-color: rgba(124,106,255,0.2); transform: translateY(-1px); }
  .card::before {
    content: ''; position: absolute; inset: 0; border-radius: 16px;
    background: var(--card-glow); opacity: 0; transition: opacity 0.3s; pointer-events: none;
  }
  .card:hover::before { opacity: 1; }

  .card-label {
    font-size: 0.65rem; letter-spacing: 0.15em; text-transform: uppercase;
    color: var(--muted); margin-bottom: 1.2rem;
    display: flex; align-items: center; gap: 0.5rem;
  }
  .card-label-dot { width: 4px; height: 4px; border-radius: 50%; background: var(--accent); }

  /* Grid */
  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }
  .grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem; }

  /* Form elements */
  .field { margin-bottom: 1.2rem; }
  .field label {
    display: block; font-size: 0.7rem; letter-spacing: 0.1em;
    color: var(--muted); margin-bottom: 0.5rem; text-transform: uppercase;
  }
  .field input, .field select, .field textarea {
    width: 100%;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.75rem 1rem;
    color: var(--text);
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    outline: none;
    transition: border-color 0.2s, box-shadow 0.2s;
  }
  .field input:focus, .field select:focus, .field textarea:focus {
    border-color: rgba(124,106,255,0.4);
    box-shadow: 0 0 0 3px rgba(124,106,255,0.08);
  }
  .field select option { background: var(--surface2); }

  /* URL input special */
  .url-wrapper { position: relative; }
  .url-prefix {
    position: absolute; left: 1rem; top: 50%; transform: translateY(-50%);
    color: var(--accent); font-size: 0.75rem; pointer-events: none;
  }
  .url-wrapper input { padding-left: 2.5rem; }

  /* Buttons */
  .btn {
    padding: 0.75rem 1.5rem;
    border-radius: 10px;
    border: none;
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    letter-spacing: 0.05em;
    cursor: pointer;
    transition: all 0.2s;
    display: inline-flex; align-items: center; gap: 0.5rem;
  }
  .btn-primary {
    background: var(--accent);
    color: white;
    box-shadow: 0 0 20px rgba(124,106,255,0.3);
  }
  .btn-primary:hover { background: #9178ff; box-shadow: 0 0 30px rgba(124,106,255,0.5); transform: translateY(-1px); }
  .btn-primary:disabled { opacity: 0.4; cursor: not-allowed; transform: none; }
  .btn-ghost {
    background: transparent;
    color: var(--muted);
    border: 1px solid var(--border);
  }
  .btn-ghost:hover { color: var(--text); border-color: rgba(255,255,255,0.15); }
  .btn-sm { padding: 0.4rem 0.9rem; font-size: 0.72rem; }

  /* Spinner */
  .spinner {
    width: 16px; height: 16px;
    border: 2px solid rgba(255,255,255,0.2);
    border-top-color: white;
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* Status bar */
  .status-bar {
    display: flex; align-items: center; gap: 1rem;
    padding: 1rem 1.5rem;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 12px;
    margin-top: 1.2rem;
    font-size: 0.78rem;
  }
  .status-icon { width: 8px; height: 8px; border-radius: 50%; }
  .status-loading { background: var(--accent); animation: pulse 1s infinite; }
  .status-success { background: var(--accent3); }
  .status-error { background: var(--accent2); }

  /* Progress bar */
  .progress-track {
    height: 2px; background: var(--border); border-radius: 2px;
    overflow: hidden; margin-top: 0.75rem;
  }
  .progress-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--accent), var(--accent3));
    border-radius: 2px;
    transition: width 0.4s ease;
  }
  .progress-indeterminate {
    height: 100%;
    background: linear-gradient(90deg, transparent, var(--accent), transparent);
    animation: indeterminate 1.5s ease infinite;
    width: 40%;
  }
  @keyframes indeterminate { 0% { transform: translateX(-150%); } 100% { transform: translateX(400%); } }

  /* Preferences section */
  .pref-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }

  .tone-chips { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.5rem; }
  .tone-chip {
    padding: 0.35rem 0.8rem;
    border-radius: 20px;
    border: 1px solid var(--border);
    background: transparent;
    color: var(--muted);
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    cursor: pointer;
    transition: all 0.2s;
    letter-spacing: 0.05em;
  }
  .tone-chip:hover { border-color: rgba(124,106,255,0.3); color: var(--text); }
  .tone-chip.selected { background: rgba(124,106,255,0.15); border-color: var(--accent); color: var(--accent); }

  /* Results */
  .results-header {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 1.5rem;
  }
  .results-title {
    font-family: 'Syne', sans-serif; font-weight: 700; font-size: 1.1rem;
    letter-spacing: -0.02em;
  }
  .count-badge {
    background: rgba(124,106,255,0.12);
    border: 1px solid rgba(124,106,255,0.2);
    color: var(--accent);
    padding: 2px 10px; border-radius: 20px; font-size: 0.7rem;
  }

  /* Post card */
  .post-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    overflow: hidden;
    transition: all 0.3s;
    animation: fadeUp 0.5s ease both;
  }
  .post-card:hover { border-color: rgba(124,106,255,0.25); transform: translateY(-2px); box-shadow: 0 12px 40px rgba(0,0,0,0.4); }
  @keyframes fadeUp { from { opacity:0; transform:translateY(16px); } to { opacity:1; transform:translateY(0); } }

  .post-img {
    width: 100%; height: 160px; object-fit: cover;
    background: linear-gradient(135deg, var(--surface2), var(--surface));
    display: flex; align-items: center; justify-content: center;
    color: var(--muted); font-size: 0.7rem; letter-spacing: 0.1em;
  }
  .post-body { padding: 1.2rem; }
  .post-title {
    font-family: 'Syne', sans-serif; font-weight: 700; font-size: 0.85rem;
    line-height: 1.4; margin-bottom: 0.6rem; letter-spacing: -0.01em;
    display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;
  }
  .post-content {
    font-size: 0.75rem; color: var(--muted); line-height: 1.6;
    margin-bottom: 1rem;
    display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden;
  }
  .post-footer {
    display: flex; align-items: center; justify-content: space-between;
    padding-top: 0.75rem;
    border-top: 1px solid var(--border);
  }
  .post-url { font-size: 0.68rem; color: var(--muted); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 180px; }

  /* Copy button */
  .copy-btn {
    padding: 0.3rem 0.7rem;
    border-radius: 6px;
    border: 1px solid var(--border);
    background: transparent;
    color: var(--muted);
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    cursor: pointer;
    transition: all 0.2s;
    letter-spacing: 0.05em;
  }
  .copy-btn:hover { border-color: var(--accent3); color: var(--accent3); }
  .copy-btn.copied { border-color: var(--accent3); color: var(--accent3); background: rgba(106,255,206,0.08); }

  /* Article card */
  .article-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem;
    animation: fadeUp 0.5s ease both;
    transition: all 0.3s;
  }
  .article-card:hover { border-color: rgba(124,106,255,0.2); }
  .article-num {
    font-size: 0.65rem; color: var(--accent); letter-spacing: 0.1em;
    margin-bottom: 0.5rem;
  }
  .article-title {
    font-family: 'Syne', sans-serif; font-weight: 700; font-size: 0.9rem;
    line-height: 1.35; margin-bottom: 0.6rem; letter-spacing: -0.01em;
  }
  .article-desc { font-size: 0.75rem; color: var(--muted); line-height: 1.65; }

  /* Analysis card */
  .analysis-pre {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    font-size: 0.78rem;
    line-height: 1.7;
    color: var(--accent3);
    white-space: pre-wrap;
    margin-top: 0.5rem;
  }

  /* Empty state */
  .empty-state {
    text-align: center; padding: 4rem 2rem;
    border: 1px dashed var(--border);
    border-radius: 16px;
    color: var(--muted);
  }
  .empty-state-icon { font-size: 2.5rem; margin-bottom: 1rem; opacity: 0.4; }
  .empty-state-text { font-size: 0.8rem; line-height: 1.6; }

  /* Tab content */
  .tab-content { animation: fadeIn 0.25s ease; }
  @keyframes fadeIn { from { opacity:0; } to { opacity:1; } }

  /* Toast */
  .toast {
    position: fixed; bottom: 2rem; right: 2rem; z-index: 100;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 0.9rem 1.4rem;
    font-size: 0.78rem;
    display: flex; align-items: center; gap: 0.75rem;
    animation: slideUp 0.3s ease;
    box-shadow: 0 8px 32px rgba(0,0,0,0.5);
  }
  @keyframes slideUp { from { opacity:0; transform:translateY(12px); } to { opacity:1; transform:translateY(0); } }
  .toast-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--accent3); flex-shrink: 0; }

  /* Divider */
  .divider { height: 1px; background: var(--border); margin: 1.5rem 0; }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-track { background: var(--bg); }
  ::-webkit-scrollbar-thumb { background: var(--surface2); border-radius: 2px; }

  @media (max-width: 768px) {
    .header { padding: 1rem 1.5rem; }
    .main { padding: 1.5rem; }
    .grid-2, .pref-grid { grid-template-columns: 1fr; }
    .grid-3 { grid-template-columns: 1fr 1fr; }
    .hero-title { font-size: 2rem; }
    .nav-tabs { display: none; }
  }
`;

const TONES = ["professional", "casual", "witty", "inspirational", "bold", "minimal"];
const LANGUAGES = [
  { code: "en", label: "English" },
  { code: "hi", label: "Hindi" },
  { code: "fr", label: "French" },
  { code: "de", label: "German" },
  { code: "es", label: "Spanish" },
  { code: "ja", label: "Japanese" },
  { code: "zh-CN", label: "Chinese" },
  { code: "ar", label: "Arabic" },
];

export default function SmartPostAI() {
  const [activeTab, setActiveTab] = useState("generate");
  const [url, setUrl] = useState("");
  const [language, setLanguage] = useState("en");
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState(null); // { type, message }
  const [posts, setPosts] = useState([]);
  const [articles, setArticles] = useState([]);
  const [preferences, setPreferences] = useState({ tone: "", topics: "", language: "en" });
  const [prefSaved, setPrefSaved] = useState(false);
  const [analysis, setAnalysis] = useState("");
  const [analysisLoading, setAnalysisLoading] = useState(false);
  const [toast, setToast] = useState(null);
  const [copiedIdx, setCopiedIdx] = useState(null);

  const showToast = (msg) => {
    setToast(msg);
    setTimeout(() => setToast(null), 3000);
  };

  const handleGenerate = async () => {
    if (!url.trim()) return;
    setLoading(true);
    setStatus({ type: "loading", message: "Scraping URLs and analysing content..." });
    setPosts([]); setArticles([]);

    try {
      const res = await fetch("/scrape_and_generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: url.trim(), language }),
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Server error");
      }
      setStatus({ type: "loading", message: "Fetching generated posts..." });

      const postsRes = await fetch("/posts.json");
      const postsData = await postsRes.json();
      setPosts(postsData);

      // Try to also fetch articles.json if server exposes it
      try {
        const artRes = await fetch("/articles.json");
        if (artRes.ok) setArticles(await artRes.json());
      } catch {}

      setStatus({ type: "success", message: `Generated ${postsData.length} posts successfully.` });
      showToast(`✓ ${postsData.length} posts generated`);
      setActiveTab("posts");
    } catch (e) {
      setStatus({ type: "error", message: e.message });
      showToast("Error: " + e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleSavePrefs = async () => {
    try {
      await fetch("/save_preferences", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...preferences, language }),
      });
      setPrefSaved(true);
      showToast("✓ Preferences saved");
      setTimeout(() => setPrefSaved(false), 2000);
    } catch (e) {
      showToast("Failed to save preferences");
    }
  };

  const handleAnalyze = async () => {
    setAnalysisLoading(true);
    try {
      const res = await fetch("/analyze_preferences");
      const data = await res.json();
      setAnalysis(data.analysis);
    } catch {
      setAnalysis("Failed to load analysis.");
    }
    setAnalysisLoading(false);
  };

  const handleCopy = (text, idx) => {
    navigator.clipboard.writeText(text);
    setCopiedIdx(idx);
    setTimeout(() => setCopiedIdx(null), 2000);
  };

  return (
    <>
      <style>{styles}</style>
      <div className="app">
        <div className="mesh mesh-1" />
        <div className="mesh mesh-2" />
        <div className="mesh mesh-3" />

        {/* Header */}
        <header className="header">
          <div className="logo">
            <div className="logo-dot" />
            SmartPost<span style={{ color: "var(--accent)", fontStyle: "italic" }}>AI</span>
          </div>
          <nav className="nav-tabs">
            {["generate", "posts", "articles", "preferences"].map(tab => (
              <button
                key={tab}
                className={`nav-tab ${activeTab === tab ? "active" : ""}`}
                onClick={() => setActiveTab(tab)}
              >
                {tab}
              </button>
            ))}
          </nav>
          <div className="badge">v2.0 · GROQ</div>
        </header>

        <main className="main">
          {/* Hero */}
          {activeTab === "generate" && (
            <div className="hero tab-content">
              <div className="hero-eyebrow">Content Intelligence Engine</div>
              <h1 className="hero-title">
                Scrape. <span>Generate.</span><br />Post Smarter.
              </h1>
              <p className="hero-sub">
                Drop any URL, pick a language — get AI-crafted summaries and social posts ready to publish.
              </p>
            </div>
          )}

          {/* Generate Tab */}
          {activeTab === "generate" && (
            <div className="tab-content">
              <div className="grid-2">
                {/* Input card */}
                <div className="card">
                  <div className="card-label"><span className="card-label-dot" /> Source URL</div>
                  <div className="field">
                    <label>Website URL</label>
                    <div className="url-wrapper">
                      <span className="url-prefix">↗</span>
                      <input
                        type="url"
                        placeholder="https://techcrunch.com"
                        value={url}
                        onChange={e => setUrl(e.target.value)}
                        onKeyDown={e => e.key === "Enter" && handleGenerate()}
                      />
                    </div>
                  </div>
                  <div className="field">
                    <label>Output Language</label>
                    <select value={language} onChange={e => setLanguage(e.target.value)}>
                      {LANGUAGES.map(l => (
                        <option key={l.code} value={l.code}>{l.label}</option>
                      ))}
                    </select>
                  </div>
                  <button
                    className="btn btn-primary"
                    style={{ width: "100%", justifyContent: "center", marginTop: "0.5rem" }}
                    onClick={handleGenerate}
                    disabled={loading || !url.trim()}
                  >
                    {loading ? <><div className="spinner" /> Processing…</> : "→ Generate Posts"}
                  </button>
                </div>

                {/* Info card */}
                <div className="card">
                  <div className="card-label"><span className="card-label-dot" /> How it works</div>
                  <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
                    {[
                      ["01", "Scrape", "Crawls same-domain URLs from your base URL"],
                      ["02", "Analyse", "Cleans and extracts meaningful content"],
                      ["03", "Generate", "AI summarises each article with your preferences"],
                      ["04", "Post", "Creates ready-to-use social media posts"],
                    ].map(([num, title, desc]) => (
                      <div key={num} style={{ display: "flex", gap: "1rem", alignItems: "flex-start" }}>
                        <span style={{ color: "var(--accent)", fontSize: "0.65rem", fontFamily: "DM Mono", minWidth: "24px", paddingTop: "2px" }}>{num}</span>
                        <div>
                          <div style={{ fontSize: "0.78rem", fontFamily: "Syne", fontWeight: 700, marginBottom: "0.2rem" }}>{title}</div>
                          <div style={{ fontSize: "0.72rem", color: "var(--muted)", lineHeight: 1.5 }}>{desc}</div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* Status */}
              {status && (
                <div className="status-bar" style={{ animationDelay: "0.1s" }}>
                  <span className={`status-icon status-${status.type}`} />
                  <span style={{ flex: 1 }}>{status.message}</span>
                  {status.type !== "loading" && (
                    <button className="btn btn-ghost btn-sm" onClick={() => setStatus(null)}>dismiss</button>
                  )}
                </div>
              )}
              {loading && (
                <div className="progress-track">
                  <div className="progress-indeterminate" />
                </div>
              )}
            </div>
          )}

          {/* Posts Tab */}
          {activeTab === "posts" && (
            <div className="tab-content">
              <div className="results-header">
                <div className="results-title">Social Media Posts</div>
                {posts.length > 0 && <span className="count-badge">{posts.length} posts</span>}
              </div>
              {posts.length === 0 ? (
                <div className="empty-state">
                  <div className="empty-state-icon">◎</div>
                  <div className="empty-state-text">No posts generated yet.<br />Head to Generate to get started.</div>
                </div>
              ) : (
                <div className="grid-3">
                  {posts.map((post, i) => (
                    <div key={i} className="post-card" style={{ animationDelay: `${i * 0.07}s` }}>
                      {post.image_url ? (
                        <img src={post.image_url} alt="" className="post-img" style={{ width: "100%", height: "160px", objectFit: "cover" }} />
                      ) : (
                        <div className="post-img">NO IMAGE</div>
                      )}
                      <div className="post-body">
                        <div className="post-content">{post.post_content}</div>
                        <div className="post-footer">
                          <span className="post-url">{post.url}</span>
                          <button
                            className={`copy-btn ${copiedIdx === i ? "copied" : ""}`}
                            onClick={() => handleCopy(post.post_content, i)}
                          >
                            {copiedIdx === i ? "copied ✓" : "copy"}
                          </button>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Articles Tab */}
          {activeTab === "articles" && (
            <div className="tab-content">
              <div className="results-header">
                <div className="results-title">Article Summaries</div>
                {articles.length > 0 && <span className="count-badge">{articles.length} articles</span>}
              </div>
              {articles.length === 0 ? (
                <div className="empty-state">
                  <div className="empty-state-icon">▣</div>
                  <div className="empty-state-text">No articles yet.<br />Generate posts first to see summaries here.</div>
                </div>
              ) : (
                <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
                  {articles.map((art, i) => (
                    <div key={i} className="article-card" style={{ animationDelay: `${i * 0.06}s` }}>
                      <div className="article-num">ARTICLE {String(i + 1).padStart(2, "0")}</div>
                      <div className="article-title">{art.title}</div>
                      <div className="article-desc">{art.description}</div>
                      <div style={{ marginTop: "0.75rem", display: "flex", alignItems: "center", gap: "1rem" }}>
                        <a href={art.url} target="_blank" rel="noreferrer"
                          style={{ fontSize: "0.7rem", color: "var(--accent)", textDecoration: "none", letterSpacing: "0.05em" }}>
                          ↗ Open Article
                        </a>
                        <button className={`copy-btn ${copiedIdx === `art-${i}` ? "copied" : ""}`}
                          onClick={() => handleCopy(art.description, `art-${i}`)}>
                          {copiedIdx === `art-${i}` ? "copied ✓" : "copy summary"}
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Preferences Tab */}
          {activeTab === "preferences" && (
            <div className="tab-content">
              <div className="results-header">
                <div className="results-title">Content Preferences</div>
              </div>
              <div className="grid-2" style={{ gap: "1.5rem" }}>
                {/* Settings */}
                <div className="card">
                  <div className="card-label"><span className="card-label-dot" /> Your Preferences</div>
                  <div className="field">
                    <label>Tone</label>
                    <div className="tone-chips">
                      {TONES.map(t => (
                        <button
                          key={t}
                          className={`tone-chip ${preferences.tone === t ? "selected" : ""}`}
                          onClick={() => setPreferences(p => ({ ...p, tone: t }))}
                        >
                          {t}
                        </button>
                      ))}
                    </div>
                  </div>
                  <div className="field" style={{ marginTop: "1rem" }}>
                    <label>Topics of Interest</label>
                    <input
                      placeholder="technology, AI, startups…"
                      value={preferences.topics}
                      onChange={e => setPreferences(p => ({ ...p, topics: e.target.value }))}
                    />
                  </div>
                  <div className="field">
                    <label>Default Language</label>
                    <select value={preferences.language} onChange={e => setPreferences(p => ({ ...p, language: e.target.value }))}>
                      {LANGUAGES.map(l => (
                        <option key={l.code} value={l.code}>{l.label}</option>
                      ))}
                    </select>
                  </div>
                  <div style={{ display: "flex", gap: "0.75rem", marginTop: "0.5rem" }}>
                    <button className="btn btn-primary btn-sm" onClick={handleSavePrefs}>
                      {prefSaved ? "✓ Saved" : "Save Preferences"}
                    </button>
                  </div>
                </div>

                {/* Analysis */}
                <div className="card">
                  <div className="card-label"><span className="card-label-dot" style={{ background: "var(--accent3)" }} /> Preference History Analysis</div>
                  <p style={{ fontSize: "0.75rem", color: "var(--muted)", lineHeight: 1.6, marginBottom: "1rem" }}>
                    Analyse your past preference history to understand your most common tones and topics.
                  </p>
                  <button className="btn btn-ghost btn-sm" onClick={handleAnalyze} disabled={analysisLoading}>
                    {analysisLoading ? <><div className="spinner" /> Analysing…</> : "→ Analyse History"}
                  </button>
                  {analysis && (
                    <div className="analysis-pre">{analysis}</div>
                  )}
                </div>
              </div>
            </div>
          )}
        </main>

        {/* Toast */}
        {toast && (
          <div className="toast">
            <span className="toast-dot" />
            {toast}
          </div>
        )}
      </div>
    </>
  );
}