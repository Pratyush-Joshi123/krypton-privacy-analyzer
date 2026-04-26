from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
from groq import Groq
import os
import json
import base64
import httpx
import sqlite3
import time
from datetime import datetime

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found! Check your .env file.")

client = genai.Client(api_key=GOOGLE_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

app = FastAPI(title="Privacy Policy Analyzer", version="5.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Database setup ───────────────────────────────────────────────────────────

DB_PATH = "history.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            app_name TEXT,
            input_type TEXT,
            score_100 INTEGER,
            score_10 REAL,
            verdict TEXT,
            result_json TEXT,
            created_at TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

def save_to_history(app_name: str, input_type: str, score_100: int, score_10: float, verdict: str, result: dict) -> int:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute(
        "INSERT INTO analyses (app_name, input_type, score_100, score_10, verdict, result_json, created_at) VALUES (?,?,?,?,?,?,?)",
        (app_name or "Unknown", input_type, score_100, score_10, verdict, json.dumps(result), datetime.now().strftime("%Y-%m-%d %H:%M"))
    )
    row_id = cur.lastrowid
    conn.commit()
    conn.close()
    return row_id

def get_all_history():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, app_name, input_type, score_100, score_10, verdict, created_at FROM analyses ORDER BY id DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_history_item(item_id: int):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM analyses WHERE id=?", (item_id,)).fetchone()
    conn.close()
    if not row:
        return None
    d = dict(row)
    d["result"] = json.loads(d["result_json"])
    del d["result_json"]
    return d

def delete_history_item(item_id: int):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM analyses WHERE id=?", (item_id,))
    conn.commit()
    conn.close()

def clear_all_history():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM analyses")
    conn.commit()
    conn.close()


# ─── Pydantic Models ──────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    text: str
    app_name: str = ""

class FollowUpRequest(BaseModel):
    text: str
    question: str

class AskRequest(BaseModel):
    question: str
    context: str = ""   # optional: app name / verdict for richer answers

class DataPoint(BaseModel):
    category: str
    collected: bool
    shared_with_third_party: bool
    can_opt_out: bool
    retention: str
    risk_level: str

class CategoryScore(BaseModel):
    name: str
    score: int
    label: str

class BenchmarkComparison(BaseModel):
    your_score: int
    industry_average: int
    best_in_class: int
    worst_in_class: int
    percentile: int
    comparison_note: str

class AnalyzeResponse(BaseModel):
    id: int                               # history ID
    score_100: int
    score_10: float
    verdict: str
    summary: list[str]
    risks: list[str]
    user_impact: list[str]
    recommendation: str
    ai_insights: list[str]
    technical_analysis: str
    simple_analysis: str
    actionable_solutions: list[str]
    follow_up_questions: list[str]
    collected_data: list[dict]
    data_table: list[DataPoint]
    category_scores: list[CategoryScore]
    benchmark: BenchmarkComparison

class FollowUpResponse(BaseModel):
    answer: str


# ─── Keyword heuristics ───────────────────────────────────────────────────────

def keyword_analysis(text: str) -> dict:
    text_lower = text.lower()
    word_count = len(text.split())

    high_risk_keywords = [
        "sell your data", "sell your information", "third party advertising",
        "share with partners", "tracking", "surveillance", "biometric",
        "location data", "without consent", "retain indefinitely",
    ]
    medium_risk_keywords = [
        "cookies", "analytics", "profiling", "behavioral", "opt-out",
        "marketing", "newsletter", "personalization", "data retention",
        "collect information",
    ]
    positive_keywords = [
        "opt-in", "your consent", "you can delete", "right to access",
        "gdpr", "ccpa", "encrypt", "anonymize", "do not sell", "data minimization",
    ]

    high_risk_hits = [kw for kw in high_risk_keywords if kw in text_lower]
    medium_risk_hits = [kw for kw in medium_risk_keywords if kw in text_lower]
    positive_hits = [kw for kw in positive_keywords if kw in text_lower]

    score = 65
    score -= len(high_risk_hits) * 10
    score -= len(medium_risk_hits) * 4
    score += len(positive_hits) * 6
    score = max(0, min(100, score))

    if score >= 75:
        verdict = "Low Risk — This policy appears user-friendly."
    elif score >= 50:
        verdict = "Moderate Risk — Some concerning clauses detected."
    else:
        verdict = "High Risk — This policy contains multiple red flags."

    summary = [
        f"Document contains approximately {word_count} words.",
        f"Detected {len(high_risk_hits)} high-risk clause(s).",
        f"Detected {len(medium_risk_hits)} medium-risk clause(s).",
        f"Detected {len(positive_hits)} user-protective clause(s).",
    ]

    risks = []
    if "sell your data" in text_lower or "sell your information" in text_lower:
        risks.append("Policy may allow selling your personal data to third parties.")
    if "tracking" in text_lower or "surveillance" in text_lower:
        risks.append("Policy mentions tracking or surveillance of user activity.")
    if "location data" in text_lower:
        risks.append("Location data collection detected.")
    if "biometric" in text_lower:
        risks.append("Biometric data usage mentioned — high sensitivity.")
    if "retain indefinitely" in text_lower:
        risks.append("Data may be retained indefinitely with no deletion policy.")
    if "third party advertising" in text_lower or "share with partners" in text_lower:
        risks.append("Data may be shared with advertising or partner networks.")
    if "cookies" in text_lower:
        risks.append("Cookie usage detected — may include tracking cookies.")
    if "behavioral" in text_lower or "profiling" in text_lower:
        risks.append("Behavioral profiling or user categorization may occur.")
    if not risks:
        risks.append("No significant risks detected from keyword analysis.")

    user_impact = []
    if score < 50:
        user_impact.append("Your data privacy is at significant risk under this policy.")
        user_impact.append("You may have limited control over how your data is used.")
    elif score < 75:
        user_impact.append("Your data may be used for targeted advertising or analytics.")
        user_impact.append("Review opt-out options carefully before agreeing.")
    else:
        user_impact.append("This policy appears to respect user rights.")
        user_impact.append("You likely retain meaningful control over your personal data.")

    if "opt-out" in text_lower:
        user_impact.append("An opt-out option is mentioned — look for it in account settings.")
    if "gdpr" in text_lower or "ccpa" in text_lower:
        user_impact.append("Policy references GDPR/CCPA compliance — stronger legal protections may apply.")

    if score >= 75:
        recommendation = (
            "This policy seems relatively safe. Still, read the full document "
            "and verify opt-in/opt-out preferences in your account settings."
        )
    elif score >= 50:
        recommendation = (
            "Proceed with caution. Consider opting out of data sharing and "
            "marketing features. Review sections on third-party data sharing closely."
        )
    else:
        recommendation = (
            "We strongly advise reviewing this policy with care before agreeing. "
            "Consider using the service minimally or seeking alternatives with "
            "stronger privacy commitments."
        )

    return {
        "score": score,
        "verdict": verdict,
        "summary": summary,
        "risks": risks,
        "user_impact": user_impact,
        "recommendation": recommendation,
    }


# ─── AI helpers ───────────────────────────────────────────────────────────────

def _call_gemini(prompt: str) -> str:
    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=prompt,
    )
    return response.text

def _call_gemini_with_image(prompt: str, image_bytes: bytes, mime: str) -> str:
    b64 = base64.b64encode(image_bytes).decode()
    contents = [
        {"role": "user", "parts": [
            {"inline_data": {"mime_type": mime, "data": b64}},
            {"text": prompt},
        ]}
    ]
    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=contents,
    )
    return response.text

def _call_groq(prompt: str) -> str:
    if not groq_client:
        raise RuntimeError("Groq not configured")
    chat = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=1024,
    )
    return chat.choices[0].message.content

def _call_ai(prompt: str) -> str:
    try:
        return _call_gemini(prompt)
    except Exception:
        return _call_groq(prompt)

def _parse_json(raw: str):
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


# ─── AI enrichment ────────────────────────────────────────────────────────────

def ai_data_table(text: str, app_name: str = "") -> list[dict]:
    app_ctx = f"App name: {app_name}." if app_name else ""
    prompt = f"""You are a privacy policy analyst. {app_ctx}
Read this privacy policy and identify every type of personal data mentioned.
Return ONLY a valid JSON array (no markdown, no extra text):
[
  {{
    "category": "Location",
    "collected": true,
    "shared_with_third_party": true,
    "can_opt_out": false,
    "retention": "2 years",
    "risk_level": "high"
  }}
]
Rules:
- category: one of: Location, Camera, Microphone, Contacts, Browsing History, Purchase History, Device ID, Email, Phone Number, Name, Age, Biometrics, Health Data, Financial Data, Search History, Social Connections, IP Address, Cookies, Usage Data, Messages
- risk_level: "high" | "medium" | "low"
- Only include data types actually mentioned or strongly implied
- Include at least 5 data types, max 15
Policy: {text[:5000]}"""
    try:
        raw = _call_gemini(prompt)
        result = _parse_json(raw)
        if isinstance(result, list):
            return result
        return []
    except Exception:
        return [
            {"category": "Usage Data", "collected": True, "shared_with_third_party": False, "can_opt_out": False, "retention": "Unknown", "risk_level": "medium"},
            {"category": "Cookies", "collected": True, "shared_with_third_party": True, "can_opt_out": False, "retention": "Unknown", "risk_level": "medium"},
        ]

def ai_category_scores(text: str, overall_score: int) -> list[dict]:
    prompt = f"""You are a privacy policy auditor. Overall privacy score: {overall_score}/100.
Score this policy across 6 categories (0-100, higher = better for user).
Return ONLY valid JSON array (no markdown):
[
  {{"name": "Data Collection", "score": 45, "label": "Collects a lot"}},
  {{"name": "Third-Party Sharing", "score": 30, "label": "Shared widely"}},
  {{"name": "User Control", "score": 70, "label": "Good controls"}},
  {{"name": "Data Retention", "score": 40, "label": "Kept too long"}},
  {{"name": "Transparency", "score": 60, "label": "Fairly clear"}},
  {{"name": "Security", "score": 65, "label": "Adequate"}}
]
Policy: {text[:4000]}"""
    try:
        raw = _call_gemini(prompt)
        result = _parse_json(raw)
        if isinstance(result, list) and len(result) == 6:
            return result
    except Exception:
        pass
    base = overall_score
    return [
        {"name": "Data Collection", "score": max(0, base - 10), "label": "Estimated"},
        {"name": "Third-Party Sharing", "score": max(0, base - 15), "label": "Estimated"},
        {"name": "User Control", "score": min(100, base + 5), "label": "Estimated"},
        {"name": "Data Retention", "score": max(0, base - 5), "label": "Estimated"},
        {"name": "Transparency", "score": min(100, base + 10), "label": "Estimated"},
        {"name": "Security", "score": base, "label": "Estimated"},
    ]

def ai_benchmark(text: str, score: int, app_name: str = "") -> dict:
    app_ctx = f"The app is called: {app_name}." if app_name else ""
    prompt = f"""You are a privacy benchmark analyst. {app_ctx}
Policy score: {score}/100. Industry average ~52/100. Best in class ~85/100. Worst ~18/100.
Return ONLY valid JSON (no markdown):
{{
  "your_score": {score},
  "industry_average": 52,
  "best_in_class": 85,
  "worst_in_class": 18,
  "percentile": 65,
  "comparison_note": "Better privacy practices than approximately 65% of apps"
}}
Policy snippet: {text[:2000]}"""
    try:
        raw = _call_gemini(prompt)
        result = _parse_json(raw)
        if isinstance(result, dict) and "percentile" in result:
            result["your_score"] = score
            return result
    except Exception:
        pass
    if score >= 75:
        return {"your_score": score, "industry_average": 52, "best_in_class": 85, "worst_in_class": 18, "percentile": 80, "comparison_note": "Better privacy practices than approximately 80% of apps"}
    elif score >= 50:
        return {"your_score": score, "industry_average": 52, "best_in_class": 85, "worst_in_class": 18, "percentile": 50, "comparison_note": "About average compared to most apps"}
    else:
        return {"your_score": score, "industry_average": 52, "best_in_class": 85, "worst_in_class": 18, "percentile": 20, "comparison_note": "Privacy practices are weaker than approximately 80% of apps"}

def ai_full_analysis(text: str, keyword_result: dict, app_name: str = "") -> dict:
    app_ctx = f"The app/service is called: {app_name}." if app_name else ""
    snippet = text[:4000]
    score = keyword_result["score"]
    risks_str = ", ".join(keyword_result["risks"])

    insights_prompt = f"""You are a privacy policy expert. {app_ctx}
Keyword analysis: Score {score}/100, risks: {risks_str}
Give 3-5 insights the keyword analysis MISSED. Focus on vague language, hidden clauses, unusual terms.
Respond ONLY as a JSON array of strings.
Policy: {snippet}"""
    try:
        ai_insights = _parse_json(_call_ai(insights_prompt))
        if not isinstance(ai_insights, list):
            ai_insights = ["AI insights could not be generated."]
    except Exception:
        ai_insights = ["AI insights could not be generated."]

    tech_prompt = f"""You are a privacy law and data security expert. {app_ctx}
Write a TECHNICAL analysis for developers and privacy professionals.
Cover: data types, retention, third-party sharing, legal bases, GDPR/CCPA gaps, tracking, security.
3-4 detailed paragraphs. Use technical terminology.
Policy: {snippet}"""
    try:
        technical_analysis = _call_ai(tech_prompt).strip()
    except Exception:
        technical_analysis = "Technical analysis unavailable."

    simple_prompt = f"""You are explaining a privacy policy to a 15-year-old. {app_ctx}
Write a SHORT plain-English summary. No jargon. Simple words. 2-3 short paragraphs.
Be honest about what's good and bad.
Policy: {snippet}"""
    try:
        simple_analysis = _call_ai(simple_prompt).strip()
    except Exception:
        simple_analysis = "Simple analysis unavailable."

    solutions_prompt = f"""You are a privacy advisor helping a non-technical user. {app_ctx}
Give 4-6 SPECIFIC, ACTIONABLE steps the user can take RIGHT NOW.
Like "Go to Settings > Privacy > disable Location" or "Disable microphone permission".
Respond ONLY as a JSON array of strings.
Policy: {snippet}"""
    try:
        actionable_solutions = _parse_json(_call_ai(solutions_prompt))
        if not isinstance(actionable_solutions, list):
            actionable_solutions = ["Review and disable unnecessary app permissions in your phone settings."]
    except Exception:
        actionable_solutions = ["Review and disable unnecessary app permissions in your phone settings."]

    followup_prompt = f"""You are a privacy expert. {app_ctx}
Read this specific policy and generate 5 follow-up questions a concerned user would want answered.
Rules:
- 3 must be SPECIFIC to what THIS policy says (e.g. if it mentions location data, ask about that specifically)
- 2 can be general but still relevant to this policy
- Questions should be short, direct, like something a real person would ask
- Do NOT use generic filler questions unrelated to the policy
Respond ONLY as a JSON array of strings. No markdown.
Policy: {snippet}"""
    try:
        follow_up_questions = _parse_json(_call_ai(followup_prompt))
        if not isinstance(follow_up_questions, list):
            raise ValueError()
    except Exception:
        follow_up_questions = [
            "What data is shared with third parties?",
            "How can I delete my account and data?",
            "Is my data sold to advertisers?",
            "How long is my data retained?",
        ]

    collected_data_prompt = f"""You are a privacy analyst. {app_ctx}
Read this policy and check which data types are collected.
Return ONLY a valid JSON array. No markdown, no extra text. Example format:
[{{"type": "IP Address", "collected": true}}, {{"type": "Camera", "collected": false}}]
Check ALL of these exactly: IP Address, Camera, Microphone, Location, Contacts, Browsing History, Device ID, Email, Phone Number, Name, Age, Biometrics, Health Data, Financial Data, Search History, Social Connections, Cookies, Usage Data, Messages, Third-Party Data Sharing
Return all 20 items with true or false based on the policy.
Policy: {snippet}"""
    try:
        collected_data = _parse_json(_call_ai(collected_data_prompt))
        if not isinstance(collected_data, list):
            raise ValueError()
    except Exception:
        collected_data = []

    return {
        "ai_insights": ai_insights,
        "technical_analysis": technical_analysis,
        "simple_analysis": simple_analysis,
        "actionable_solutions": actionable_solutions,
        "follow_up_questions": follow_up_questions,
        "collected_data": collected_data,
    }


# ─── Text extractors ──────────────────────────────────────────────────────────

def extract_text_from_url(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = httpx.get(url, headers=headers, timeout=15, follow_redirects=True)
    resp.raise_for_status()
    import re
    text = re.sub(r'<[^>]+>', ' ', resp.text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:8000]

def extract_text_from_image(image_bytes: bytes, mime: str) -> str:
    prompt = "Extract ALL visible text from this image exactly as written. Return only the text, no commentary."
    return _call_gemini_with_image(prompt, image_bytes, mime)

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    import io
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(pdf_bytes))
        return " ".join(page.extract_text() or "" for page in reader.pages)[:8000]
    except Exception as e:
        return f"PDF extraction failed: {e}"


# ─── Core pipeline ────────────────────────────────────────────────────────────

def full_pipeline(text: str, app_name: str = "", input_type: str = "text") -> dict:
    kw = keyword_analysis(text)
    ai = ai_full_analysis(text, kw, app_name)
    score_100 = kw["score"]
    score_10 = round(score_100 / 10, 1)
    data_table = ai_data_table(text, app_name)
    category_scores = ai_category_scores(text, score_100)
    benchmark = ai_benchmark(text, score_100, app_name)

    result = {
        "score_100": score_100,
        "score_10": score_10,
        "verdict": kw["verdict"],
        "summary": kw["summary"],
        "risks": kw["risks"],
        "user_impact": kw["user_impact"],
        "recommendation": kw["recommendation"],
        **ai,
        "data_table": data_table,
        "category_scores": category_scores,
        "benchmark": benchmark,
    }

    # Save to history
    history_id = save_to_history(app_name, input_type, score_100, score_10, kw["verdict"], result)
    result["id"] = history_id
    return result


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "Privacy Policy Analyzer v5 — with History"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest):
    """Analyze pasted privacy policy text."""
    return full_pipeline(request.text, request.app_name, input_type="text")


@app.post("/analyze-url", response_model=AnalyzeResponse)
def analyze_url(url: str = Form(...), app_name: str = Form("")):
    """Fetch a URL and analyze its privacy policy."""
    text = extract_text_from_url(url)
    return full_pipeline(text, app_name, input_type="url")


@app.post("/analyze-file", response_model=AnalyzeResponse)
async def analyze_file(
    file: UploadFile = File(...),
    app_name: str = Form(""),
):
    """Analyze a privacy policy from a PDF or screenshot."""
    content = await file.read()
    mime = file.content_type or ""
    if "pdf" in mime:
        text = extract_text_from_pdf(content)
        input_type = "pdf"
    elif mime.startswith("image/"):
        text = extract_text_from_image(content, mime)
        input_type = "screenshot"
    else:
        return {"error": "Unsupported file type. Upload a PDF or image."}
    return full_pipeline(text, app_name, input_type=input_type)


@app.post("/followup", response_model=FollowUpResponse)
def follow_up(request: FollowUpRequest):
    """Answer a follow-up question about a privacy policy."""
    prompt = f"""You are a privacy policy expert. Answer clearly in 2-3 sentences. Use plain English.
Policy (first 3000 chars): {request.text[:3000]}
User question: {request.question}"""
    try:
        answer = _call_ai(prompt).strip()
    except Exception as e:
        answer = f"Could not answer: {e}"
    return {"answer": answer}


@app.post("/ask")
def ask_question(request: AskRequest):
    """Answer a plain privacy/tech question in simple language."""
    ctx = f" The user has just analysed the privacy policy of {request.context}." if request.context else ""
    prompt = f"""You are Krypton's privacy assistant.{ctx}
Answer the following question in very simple, friendly language — like explaining to a curious friend, not a textbook.
Keep it concise (2-4 short paragraphs max). No jargon. Get straight to the point.
Question: {request.question}"""
    try:
        answer = _call_ai(prompt).strip()
    except Exception as e:
        answer = f"Could not answer: {e}"
    return {"answer": answer}


# ─── History Routes ───────────────────────────────────────────────────────────

@app.get("/history")
def get_history():
    """Get all past analyses (summary only, no full result)."""
    return get_all_history()


@app.get("/history/{item_id}")
def get_history_by_id(item_id: int):
    """Get full result of a past analysis by ID."""
    item = get_history_item(item_id)
    if not item:
        return {"error": "Not found"}
    return item


@app.delete("/history/{item_id}")
def delete_history(item_id: int):
    """Delete a single history item."""
    delete_history_item(item_id)
    return {"message": "Deleted"}


@app.delete("/history")
def clear_history():
    """Clear all history."""
    clear_all_history()
    return {"message": "History cleared"}