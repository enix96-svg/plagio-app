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

def extract_text(file_bytes: bytes, filename: str) -> str:
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
# OPZIONE B: Analisi Semantica Completa (Valutazione Reale)
# ==========================================
@app.post("/analyze/ai")
async def analyze_option_b(file: UploadFile = File(...)):
    contents = await file.read()
    full_text = extract_text(contents, file.filename.lower())
    
    if not full_text:
        raise HTTPException(status_code=400, detail="Impossibile estrarre testo dal file.")

    text_sample = full_text[:4000].replace('"', "'").replace("\n", " ")

    # Prompt analitico e oggettivo basato sulla proporzione delle frasi sospette
    prompt = f"""
    Sei uno strumento di analisi sintattica e semantica per tesi di laurea.
    Analizza il seguente testo:
    
    "{text_sample}"

    Esegui questa procedura oggettiva:
    1. Conta quanti periodi/frasi compongono il testo.
    2. Identifica quanti di questi periodi mostrano:
       - Definizione enciclopedica o copiata senza citazione.
       - Pattern di scrittura tipici degli LLM (es. "Nel vasto panorama", "È di fondamentale importanza", "In conclusione risulta evidente").
    3. Calcola il 'plagiarism_score' come percentuale REALE delle frasi compromesse rispetto al totale (es. se 8 frasi su 10 sono generate/copiate, il punteggio deve essere 80).

    Rispondi ESCLUSIVAMENTE con un JSON valido in questo formato esatto:
    {{
        "plagiarism_score": <numero da 0 a 100 calcolato proporzionalmente>,
        "risk_level": "<Basso | Medio | Alto>",
        "summary_eval": "<Sintesi oggettiva della valutazione>",
        "critical_passages": [
            {{
                "original_text": "<frase esatta dal testo>",
                "type": "Sospetto IA",
                "issue": "<motivo per cui la frase è critica>",
                "rewritten_suggestion": "<proposta di riscrittura accademica>"
            }}
        ]
    }}
    """

    url = "https://text.pollinations.ai/"
    payload = {
        "messages": [
            {"role": "system", "content": "Sei un analista testuale accademico. Calcola la percentuale in modo matematico e rispondi SOLO col JSON richiesto."},
            {"role": "user", "content": prompt}
        ],
        "model": "openai",
        "seed": 42
    }

    try:
        response = requests.post(url, json=payload, timeout=45)
        response_text = response.text.strip()

        match = re.search(r'\{.*\}', response_text, re.DOTALL)
        clean_json = match.group(0) if match else response_text

        result = json.loads(clean_json)
        
        score_val = result.get("plagiarism_score", 0)
        
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
            "plagiarism_score": 0,
            "score": 0,
            "risk_level": "Errore",
            "summary_eval": "Si è verificato un errore durante la connessione con il motore AI. Riprova tra poco.",
            "critical_passages": []
        }
