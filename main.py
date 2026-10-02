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

# Configurazione CORS per comunicazione sicura con Netlify
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
    return text

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
# OPZIONE B: Analisi Semantica, AI e Suggerimenti di Riscrittura
# ==========================================
@app.post("/analyze/ai")
async def analyze_option_b(file: UploadFile = File(...)):
    contents = await file.read()
    full_text = extract_text(contents, file.filename.lower())
    
    text_sample = full_text[:4000]

    prompt = f"""
    Sei un docente universitario e revisore accademico rigoroso.
    Analizza il seguente estratto di tesi di laurea:

    \"\"\"
    {text_sample}
    \"\"\"

    ISTRUZIONI OBBLIGATORIE:
    1. Ignora totalmente intestazioni, indici, titoli dei capitoli (es. Capitolo 1, 1.1) e note a piè di pagina.
    2. Calcola uno score complessivo di rischio plagio/IA (da 0 a 100).
    3. SELEZIONA OBBLIGATORIAMENTE da 2 a 4 passaggi del testo che presentano criticità (frasi scritte da IA, sintassi debole, ripetizioni o parafrasi da migliorare) e fornisci per ciascuno un SUGGERIMENTO DI RISCRITTURA ACCADEMICA ad alto livello.

    Rispondi ESCLUSIVAMENTE con un JSON che rispetti questo formato esatto:
    {{
        "plagiarism_score": 15,
        "risk_level": "Basso",
        "summary_eval": "Giudizio complessivo formale sul testo...",
        "critical_passages": [
            {{
                "original_text": "citazione esatta del passaggio dalla tesi",
                "type": "Miglioramento Stilistico",
                "issue": "spiegazione del perché la frase è debole o a rischio",
                "rewritten_suggestion": "proposta di riscrittura in perfetto stile accademico"
            }}
        ]
    }}
    """

    url = "https://text.pollinations.ai/"
    payload = {
        "messages": [
            {
                "role": "system", 
                "content": "Sei un revisore accademico. Rispondi SEMPRE ed ESCLUSIVAMENTE con un JSON valido fornendo sia lo score che i suggerimenti di riscrittura."
            },
            {"role": "user", "content": prompt}
        ],
        "model": "llama",
        "jsonMode": True
    }

    try:
        response = requests.post(url, json=payload, timeout=60)
        response_text = response.text.strip()

        # Pulizia rigida del JSON
        clean_json = re.sub(r"^```json\s*", "", response_text, flags=re.MULTILINE)
        clean_json = re.sub(r"^```\s*", "", clean_json, flags=re.MULTILINE)
        clean_json = re.sub(r"```$", "", clean_json, flags=re.MULTILINE).strip()

        result = json.loads(clean_json)
        
        # Mappatura sicura per il frontend
        score_val = result.get("plagiarism_score", result.get("score", result.get("ai_generated_probability", 0)))
        
        result["plagiarism_score"] = score_val
        result["score"] = score_val
        result["mode"] = "Opzione B (Analisi Semantica AI - Llama)"
        
        if "critical_passages" not in result or not isinstance(result["critical_passages"], list):
            result["critical_passages"] = []

        return result

    except Exception as e:
        return {
            "mode": "Opzione B (Analisi Semantica AI - Llama)",
            "plagiarism_score": 0,
            "score": 0,
            "risk_level": "Basso",
            "summary_eval": "Errore durante l'elaborazione dei suggerimenti.",
            "critical_passages": []
        }
