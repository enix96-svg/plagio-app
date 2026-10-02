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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def extract_clean_text(file_bytes: bytes, filename: str) -> str:
    raw_text = ""
    try:
        if filename.endswith(".docx"):
            doc = docx.Document(io.BytesIO(file_bytes))
            raw_text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        elif filename.endswith(".pdf"):
            reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    raw_text += t + "\n"
        else:
            raise HTTPException(status_code=400, detail="Formato non supportato. Carica un file PDF o DOCX.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Errore nella lettura del file: {str(e)}")

    lines = raw_text.split("\n")
    filtered_lines = []
    
    for line in lines:
        l = line.strip()
        if not l:
            continue
        if re.search(r'\.{2,}\s*\d+', l) or re.search(r'\.{4,}', l):
            continue
        if l.lower() in ["indice", "sommario", "table of contents", "executive summary"]:
            continue
            
        filtered_lines.append(l)

    return "\n".join(filtered_lines).strip()

@app.get("/")
def read_root():
    return {"status": "online", "message": "Backend Active"}

@app.post("/analyze/search")
async def analyze_option_a(file: UploadFile = File(...)):
    contents = await file.read()
    full_text = extract_clean_text(contents, file.filename.lower())
    
    paragraphs = [p.strip() for p in full_text.split("\n") if len(p.split()) >= 10]
    if not paragraphs:
        return {"error": "Testo insufficiente per effettuare l'analisi."}

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
        "score": plagiarism_score,
        "matches": matches
    }

@app.post("/analyze/ai")
async def analyze_option_b(file: UploadFile = File(...)):
    contents = await file.read()
    full_text = extract_clean_text(contents, file.filename.lower())
    
    if not full_text:
        raise HTTPException(status_code=400, detail="Impossibile estrarre testo dal file.")

    # Legge la chiave
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not gemini_key:
        return {
            "mode": "Opzione B",
            "plagiarism_score": 0,
            "score": 0,
            "risk_level": "Errore",
            "summary_eval": "ERRORE: La variabile GEMINI_API_KEY non è impostata su Render.",
            "critical_passages": []
        }

    all_paragraphs = [p.strip() for p in full_text.split("\n") if len(p.strip().split()) >= 10]
    if not all_paragraphs:
        return {
            "mode": "Opzione B",
            "plagiarism_score": 0,
            "score": 0,
            "risk_level": "Basso",
            "summary_eval": "Testo troppo breve.",
            "critical_passages": []
        }

    sample_paragraphs = all_paragraphs[:10]
    text_sample = "\n---\n".join(sample_paragraphs).replace('"', "'")

    prompt = f"""
    Sei un revisore accademico. Analizza i seguenti paragrafi:
    {text_sample}

    Rispondi SOLO in formato JSON:
    {{
        "summary_eval": "<sintesi>",
        "critical_passages": [
            {{
                "original_text": "<testo>",
                "type": "Sospetto IA",
                "issue": "<motivo>",
                "rewritten_suggestion": "<riscrittura>"
            }}
        ]
    }}
    """

    # Endpoint REST diretto senza SDK
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={gemini_key}"
    
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "response_mime_type": "application/json",
            "temperature": 0.1
        }
    }

    try:
        res = requests.post(url, json=payload, timeout=20)
        
        # Se la risposta NON è HTTP 200 OK, mostriamo subito cosa dice Google
        if res.status_code != 200:
            return {
                "mode": "Opzione B",
                "plagiarism_score": 0,
                "score": 0,
                "risk_level": "Errore API",
                "summary_eval": f"Errore da Google (HTTP {res.status_code}): {res.text}",
                "critical_passages": []
            }

        res_data = res.json()
        raw_text = res_data["candidates"][0]["content"]["parts"][0]["text"]
        result = json.loads(raw_text)

        critical = result.get("critical_passages", [])
        if not isinstance(critical, list):
            critical = []

        num_critical = len(critical)
        calculated_score = min(100, round((num_critical / len(sample_paragraphs)) * 100))
        
        risk = "Basso" if calculated_score <= 20 else ("Medio" if calculated_score <= 50 else "Alto")

        return {
            "mode": "Opzione B (Gemini API)",
            "plagiarism_score": calculated_score,
            "score": calculated_score,
            "risk_level": risk,
            "summary_eval": result.get("summary_eval", "Analisi completata."),
            "critical_passages": critical
        }

    except Exception as e:
        return {
            "mode": "Opzione B",
            "plagiarism_score": 0,
            "score": 0,
            "risk_level": "Errore Python",
            "summary_eval": f"Errore interno Python: {str(e)}",
            "critical_passages": []
        }
