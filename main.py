import os
import io
import json
import re
import docx
import pypdf
import requests
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from duckduckgo_search import DDGS

app = FastAPI(title="Real Anti-Plagiarism API")

# Configurazione CORS per comunicare con Netlify
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def extract_text(file_bytes: bytes, filename: str) -> str:
    text = ""
    if filename.endswith(".docx"):
        doc = docx.Document(io.BytesIO(file_bytes))
        text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    elif filename.endswith(".pdf"):
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    else:
        raise HTTPException(status_code=400, detail="Formato non supportato. Usa PDF o DOCX.")
    return text

@app.get("/")
def read_root():
    return {"status": "online", "message": "Backend Antiplagio Attivo"}

# ==========================================
# OPZIONE A: Ricerca Reale Fonti Web (DuckDuckGo)
# ==========================================
@app.post("/analyze/search")
async def analyze_option_a(file: UploadFile = File(...)):
    contents = await file.read()
    full_text = extract_text(contents, file.filename.lower())
    
    paragraphs = [p.strip() for p in full_text.split("\n") if len(p.split()) >= 10]
    if not paragraphs:
        return {"error": "Testo insufficiente per l'analisi."}

    matches = []
    ddgs = DDGS()
    sample_paragraphs = paragraphs[:15] 

    for p in sample_paragraphs:
        words = p.split()[:12]
        query = f'"{" ".join(words)}"'
        
        try:
            results = list(ddgs.text(query, max_results=1))
            if results:
                matches.append({
                    "excerpt": p[:150] + "...",
                    "matched_text": words,
                    "source_title": results[0].get("title", "Fonte Web"),
                    "source_url": results[0].get("href", "")
                })
        except Exception:
            continue

    plagiarism_score = min(100, round((len(matches) / len(sample_paragraphs)) * 100, 1))

    return {
        "mode": "Opzione A (Ricerca Reale Web)",
        "filename": file.filename,
        "total_paragraphs_analyzed": len(sample_paragraphs),
        "plagiarism_score": plagiarism_score,
        "matches": matches
    }

# ==========================================
# OPZIONE B: Analisi Semantica AI (Gratuita / No Key)
# ==========================================
@app.post("/analyze/ai")
async def analyze_option_b(file: UploadFile = File(...)):
    contents = await file.read()
    full_text = extract_text(contents, file.filename.lower())
    
    text_sample = full_text[:4000]

    prompt = f"""
    Sei un revisore accademico esperto in anti-plagio e stile di tesi universitarie.
    Analizza il seguente estratto di tesi:

    "{text_sample}"

    Fornisci UNA SOLA RISPOSTA in formato JSON valido, SENZA MARKDOWN, SENZA BLOCCHI CODE, usando questa struttura esatta:
    {{
        "plagiarism_score": 15,
        "risk_level": "Basso",
        "critical_passages": [
            {{
                "original_text": "frase estratta",
                "issue": "spiegazione del rischio",
                "rewritten_suggestion": "suggerimento di riscrittura"
            }}
        ]
    }}
    """

    # Endpoint AI pubblico e gratuito senza API Key
    url = "https://text.pollinations.ai/"
    payload = {
        "messages": [
            {"role": "system", "content": "Sei un'API che risponde esclusivamente in JSON valido, senza testo introduttivo o formattazione markdown."},
            {"role": "user", "content": prompt}
        ],
        "jsonMode": True
    }

    try:
        response = requests.post(url, json=payload, timeout=60)
        response_text = response.text.strip()

        # Pulizia da eventuale markdown residuo
        clean_json = re.sub(r"^```json\s*", "", response_text, flags=re.MULTILINE)
        clean_json = re.sub(r"^```\s*", "", clean_json, flags=re.MULTILINE)
        clean_json = re.sub(r"```$", "", clean_json, flags=re.MULTILINE).strip()

        result = json.loads(clean_json)
        result["mode"] = "Opzione B (Analisi Semantica AI - Free)"
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore analisi AI gratuita: {str(e)}")
