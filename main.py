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
    """Estrae il testo ed elimina Indici/Sommari per evitare falsi positivi."""
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
        # Ignora righe dell'indice
        if re.search(r'\.{2,}\s*\d+', l) or re.search(r'\.{4,}', l):
            continue
        if l.lower() in ["indice", "sommario", "table of contents", "executive summary"]:
            continue
            
        filtered_lines.append(l)

    return "\n".join(filtered_lines).strip()

@app.get("/")
def read_root():
    return {"status": "online", "message": "Backend Antiplagio Attivo e Operativo con Gemini"}

# ==========================================
# OPZIONE A: Ricerca Reale Fonti Web (DuckDuckGo)
# ==========================================
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

# ==========================================
# OPZIONE B: Analisi Semantica Potenziata con Google Gemini
# ==========================================
@app.post("/analyze/ai")
async def analyze_option_b(file: UploadFile = File(...)):
    contents = await file.read()
    full_text = extract_clean_text(contents, file.filename.lower())
    
    if not full_text:
        raise HTTPException(status_code=400, detail="Impossibile estrarre testo dal file.")

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

    # Leggiamo paragrafi di almeno 5 parole per catturare più porzioni di testo
    all_paragraphs = [p.strip() for p in full_text.split("\n") if len(p.strip().split()) >= 5]
    if not all_paragraphs:
        return {
            "mode": "Opzione B",
            "plagiarism_score": 0,
            "score": 0,
            "risk_level": "Basso",
            "summary_eval": "Il testo contiene solo titoli o frasi troppo brevi per essere analizzate.",
            "critical_passages": []
        }

    # Analizziamo fino a 20 paragrafi
    sample_paragraphs = all_paragraphs[:20]
    text_sample = "\n---\n".join(sample_paragraphs).replace('"', "'")

    prompt = f"""
    Sei un severo software antiplagio universitario ed esperto di rilevamento testi IA. Analizza attentamente questi paragrafi:

    {text_sample}

    COMPITO:
    Individua qualsiasi passaggio che sembri scritto da un'Intelligenza Artificiale (struttura rigida, parole chiave ripetitive, elenchi standardizzati) o che presenti chiari segni di plagio/parafrasi da internet. Sii rigoroso nell'analisi.

    Rispondi ESCLUSIVAMENTE in formato JSON con questa struttura esatta:
    {{
        "summary_eval": "<Un giudizio sintetico e diretto sulla qualità del testo e sul rischio rilevato>",
        "critical_passages": [
            {{
                "original_text": "<il testo esatto del passaggio sospetto>",
                "type": "Sospetto IA o Parafrasi",
                "issue": "<perché questo passaggio è considerato anomalo>",
                "rewritten_suggestion": "<come riformularlo in modo accademico originale>"
            }}
        ]
    }}
    """

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={gemini_key}"
    
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "response_mime_type": "application/json",
            "temperature": 0.4
        }
    }

    try:
        res = requests.post(url, json=payload, timeout=25)
        
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
        total_analyzed = len(sample_paragraphs)
        calculated_score = min(100, round((num_critical / total_analyzed) * 100))
        
        if calculated_score <= 20:
            risk = "Basso"
        elif calculated_score <= 50:
            risk = "Medio"
        else:
            risk = "Alto"

        return {
            "mode": "Opzione B (Analisi Semantica con Gemini)",
            "plagiarism_score": calculated_score,
            "score": calculated_score,
            "risk_level": risk,
            "summary_eval": result.get("summary_eval", "Analisi completata con successo."),
            "critical_passages": critical
        }

    except Exception as e:
        return {
            "mode": "Opzione B",
            "plagiarism_score": 0,
            "score": 0,
            "risk_level": "Errore Interno",
            "summary_eval": f"Errore durante l'elaborazione: {str(e)}",
            "critical_passages": []
        }
