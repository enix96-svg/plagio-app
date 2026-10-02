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

# Configurazione CORS per comunicazione con Netlify
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
        "matches": matches
    }

# ==========================================
# OPZIONE B: Analisi Semantica & AI (Llama 3.3 70B Fisso)
# ==========================================
@app.post("/analyze/ai")
async def analyze_option_b(file: UploadFile = File(...)):
    contents = await file.read()
    full_text = extract_text(contents, file.filename.lower())
    
    text_sample = full_text[:4000]

    prompt = f"""
    Sei un docente universitario rigoroso ed esperto in valutazione di tesi di laurea.
    Analizza il seguente estratto di tesi:

    \"\"\"
    {text_sample}
    \"\"\"

    ISTRUZIONI DI VALUTAZIONE:
    1. Ignora totalmente intestazioni, indice, numeri di capitolo (es. 1.1, Capitolo 2) e note bibliografiche.
    2. Cerca frasi generatrici da IA (es. connettivi rigidi, stile piatti tipico di ChatGPT), sintassi tradotta letteralmente o parafrasi superficiali.
    3. Assegna un punteggio percentuale reale e coerente. Se trovi passaggi sospetti, elenvali chiaramente.

    Rispondi ESCLUSIVAMENTE in formato JSON con questo schema esatto:
    {{
        "plagiarism_score": <numero intero da 0 a 100>,
        "ai_generated_probability": <numero intero da 0 a 100>,
        "risk_level": "<Basso | Medio | Alto>",
        "summary_eval": "<valutazione sintetica formale in 2 frasi>",
        "critical_passages": [
            {{
                "original_text": "<citazione esatta>",
                "type": "<Sospetto IA | Parafrasi Superficiale | Sintassi Debole>",
                "issue": "<spiegazione del problema>",
                "rewritten_suggestion": "<riscrittura accademica>"
            }}
        ]
    }}
    """

    url = "https://text.pollinations.ai/"
    payload = {
        "messages": [
            {
                "role": "system", 
                "content": "Sei un revisore accademico. Rispondi solo in formato JSON valido, senza markdown, senza blocchi di codice e senza introduzioni."
            },
            {"role": "user", "content": prompt}
        ],
        "model": "llama",
        "jsonMode": True
    }

    try:
        response = requests.post(url, json=payload, timeout=60)
        response_text = response.text.strip()

        # Pulizia rigida da marcatori markdown
        clean_json = re.sub(r"^```json\s*", "", response_text, flags=re.MULTILINE)
        clean_json = re.sub(r"^```\s*", "", clean_json, flags=re.MULTILINE)
        clean_json = re.sub(r"```$", "", clean_json, flags=re.MULTILINE).strip()

        result = json.loads(clean_json)
        result["mode"] = "Opzione B (Analisi Semantica AI - Llama)"
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore analisi AI: {str(e)}")
