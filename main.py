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

# 1. Inizializzazione dell'app FastAPI
app = FastAPI(title="Real Anti-Plagiarism API")

# 2. Configurazione CORS per Netlify / Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def extract_text(file_bytes: bytes, filename: str) -> str:
    """Estrae il testo pulito da file .docx o .pdf"""
    text = ""
    try:
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
            raise HTTPException(status_code=400, detail="Formato non supportato. Carica un file PDF o DOCX.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Errore nella lettura del file: {str(e)}")
    return text.strip()

@app.get("/")
def read_root():
    return {"status": "online", "message": "Backend Antiplagio Attivo e Operativo"}

# ==========================================
# OPZIONE A: Ricerca Reale Fonti Web (DuckDuckGo)
# ==========================================
@app.post("/analyze/search")
async def analyze_option_a(file: UploadFile = File(...)):
    contents = await file.read()
    full_text = extract_text(contents, file.filename.lower())
    
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
# OPZIONE B: Analisi Semantica Dinamica e Reale (IA & Plagio)
# ==========================================
@app.post("/analyze/ai")
async def analyze_option_b(file: UploadFile = File(...)):
    contents = await file.read()
    full_text = extract_text(contents, file.filename.lower())
    
    if not full_text:
        raise HTTPException(status_code=400, detail="Impossibile estrarre testo dal file.")

    text_sample = full_text[:4000].replace('"', "'").replace("\n", " ").replace("\\", "")

    # Prompt calibrato per punteggio oggettivo e dinamico
    prompt = f"""Analizza questo estratto di tesi e determina il punteggio reale (da 0 a 100) di plagio o testo generato da IA.

    TESTO DA ANALIZZARE:
    "{text_sample}"

    REGOLE DI CALCOLO RIGOROSE:
    - 0-20: Testo totalmente originale, stile accademico umano genuino e citazioni corrette.
    - 21-50: Testo con qualche frase generica o parafrasi lieve, ma prevalentemente originale.
    - 51-79: Testo fortemente sospetto, ricco di costrutti tipici da ChatGPT o fonti enciclopediche.
    - 80-100: Testo palesemente copiato parola per parola da fonti famose o generato integralmente da un LLM.

    Rispondi SOLTANTO con un JSON valido strutturato esattamente così (calcola il punteggio reale al posto del valore di esempio):
    {{
        "plagiarism_score": 0,
        "risk_level": "Basso",
        "summary_eval": "Spiegazione sintetica ed oggettiva della valutazione...",
        "critical_passages": [
            {{
                "original_text": "citazione della frase critica dal testo",
                "type": "Sospetto IA",
                "issue": "motivo del rischio rilevato",
                "rewritten_suggestion": "proposta di riscrittura accademica"
            }}
        ]
    }}
    """

    url = "https://text.pollinations.ai/"
    payload = {
        "messages": [
            {"role": "system", "content": "Sei un analista testuale accademico imparziale. Rispondi ESCLUSIVAMENTE con il JSON richiesto senza altro testo."},
            {"role": "user", "content": prompt}
        ],
        "model": "openai",
        "seed": 42
    }

    try:
        response = requests.post(url, json=payload, timeout=45)
        response.raise_for_status()
        response_text = response.text.strip()

        # Estrazione sicura del blocco JSON
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}')
        
        if start_idx != -1 and end_idx != -1:
            clean_json = response_text[start_idx:end_idx+1]
        else:
            clean_json = response_text

        result = json.loads(clean_json)
        
        # Converte il punteggio in numero intero
        score_val = int(result.get("plagiarism_score", 0))
        
        result["plagiarism_score"] = score_val
        result["score"] = score_val
        result["mode"] = "Opzione B (Analisi Semantica & IA)"
        
        if "critical_passages" not in result or not isinstance(result["critical_passages"], list):
            result["critical_passages"] = []

        return result

    except Exception as e:
        print(f"Errore chiamata AI: {e}")
        return {
            "mode": "Opzione B (Analisi Semantica & IA - Errore API)",
            "plagiarism_score": -1,
            "score": -1,
            "risk_level": "Errore",
            "summary_eval": f"Errore durante la connessione con l'AI: {str(e)[:60]}",
            "critical_passages": []
        }
