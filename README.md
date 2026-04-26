# Krypton — Privacy Policy Analyzer

> Know what you're actually agreeing to.

Krypton uses Gemini AI to instantly analyze any privacy policy and tell you exactly what data an app collects, how risky it is, and what you can do about it.

## Features
- Paste text, enter a URL, upload a PDF, or screenshot
- AI-powered analysis using Google Gemini
- Data collection table (what's collected, shared, retention period)
- Industry benchmark comparison
- Category scores (Data Collection, Transparency, Security etc.)
- Simple English + Technical analysis
- Actionable steps to protect your privacy
- History saved locally
- Ask follow-up questions about any policy

## How to Run
1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Create a `.env` file and add: `GOOGLE_API_KEY=your_key_here`
4. Start backend: `uvicorn main:app --reload`
5. Open frontend: `python -m http.server 3000` then go to `http://localhost:3000`

## Tech Stack
- FastAPI + Python
- Google Gemini AI
- SQLite for history
- Vanilla HTML/CSS/JS

## Track
Built for AAYAM 2026 — Track 2: Build Something Using AI
