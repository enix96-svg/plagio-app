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

# Header per simulare un browser ed evitare i blocchi anti-bot da Render
HEADERS_BROWSER = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Content-Type": "application/json"
}

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
        # Ignora righe dell'indice (punti sospensivi con numeri o parole chiave)
        if re.search(r'\.{2,}\s*\d+', l) or re.search(r'\.{4,}', l):
            continue
        if l.lower() in ["indice", "sommario", "table of contents", "executive summary"]:
            continue
            
        filtered_lines.append(l)

    return "\n".join(filtered_lines).strip()

@app.get("/")
def read_root():
    return {"status": "online", "message": "Backend Antiplagio Attivo e Operativo"}

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
# OPZIONE B: Analisi Semantica Gratis con Anti-Block Header & Fallback
# ==========================================
@app.post("/analyze/ai")
async def analyze_option_b(file: UploadFile = File(...)):
    contents = await file.read()
    full_text = extract_clean_text(contents, file.filename.lower())
    
    if not full_text:
        raise HTTPException(status_code=400, detail="Impossibile estrarre testo dal file.")

    # Analizziamo solo paragrafi veri (almeno 10 parole, escludendo titoli isolati)
    all_paragraphs = [p.strip() for p in full_text.split("\n") if len(p.strip().split()) >= 10]

    if not all_paragraphs:
        return {
            "mode": "Opzione B (Analisi Semantica & IA)",
            "plagiarism_score": 0,
            "score": 0,
            "risk_level": "Basso",
            "summary_eval": "Il testo contiene solo titoli o frasi troppo brevi per valutare il rischio.",
            "critical_passages": []
        }

    # Analizziamo fino a 8 paragrafi per rendere la risposta dell'AI più veloce
    sample_paragraphs = all_paragraphs[:8]
    text_sample = "\n---\n".join(sample_paragraphs).replace('"', "'")

    prompt = f"""
    Sei un revisore accademico esperto ed equo. Analizza i seguenti paragrafi tratti da una tesi:

    {text_sample}

    REGOLE TASSATIVE:
    1. Segnala SOLO i paragrafi che mostrano evidenze schiaccianti di testo generato da IA (stile ChatGPT meccanico e generico) o plagio integrale.
    2. IGNORA DEL TUTTO titoli, sottotitoli o formule introduttive formali accademiche.
    3. Se un paragrafo è normale testo accademico umano, NON inserirlo nei passaggi critici.

    Rispondi ESCLUSIVAMENTE con un JSON in questo formato esatto:
    {{
        "summary_eval": "<Sintesi concisa ed obiettiva dell'analisi>",
        "critical_passages": [
            {{
                "original_text": "<estratto esatto del solo paragrafo/frase realmente critico>",
                "type": "Sospetto IA",
                "issue": "<spiegazione del motivo del rischio>",
                "rewritten_suggestion": "<proposta di riscrittura accademica>"
            }}
        ]
    }}
    """

    # Tentativo 1: Chiamata POST con User-Agent
    url = "https://text.pollinations.ai/"
    payload = {
        "messages": [
            {"role": "system", "content": "Sei un analista testuale accademico. Rispondi ESCLUSIVAMENTE con il JSON richiesto."},
            {"role": "user", "content": prompt}
        ],
        "model": "openai",
        "seed": 42
    }

    response_text = ""
    try:
        response = requests.post(url, json=payload, headers=HEADERS_BROWSER, timeout=30)
        response.raise_for_status()
        response_text = response.text.strip()
    except Exception as e:
        print(f"[LOG ERROR] Tentativo POST fallito: {e}")
        # Tentativo 2: Fallback su modello Mistral se OpenAI è intasato
        try:
            payload["model"] = "mistral"
            response = requests.post(url, json=payload, headers=HEADERS_BROWSER, timeout=30)
            response.raise_for_status()
            response_text = response.text.strip()
        except Exception as e2:
            print(f"[LOG ERROR] Tentativo Fallback Mistral fallito: {e2}")
            return {
                "mode": "Opzione B (Analisi Semantica & IA - Errore API)",
                "plagiarism_score": -1,
                "score": -1,
                "risk_level": "Errore",
                "summary_eval": "Il server AI remoto è temporaneamente non raggiungibile. Riprova tra poco.",
                "critical_passages": []
            }

    try:
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}')
        
        if start_idx != -1 and end_idx != -1:
            clean_json = response_text[start_idx:end_idx+1]
        else:
            clean_json = response_text

        result = json.loads(clean_json)
        
        critical = result.get("critical_passages", [])
        if not isinstance(critical, list):
            critical = []

        # CALCOLO MATEMATICO IN PYTHON (Garantisce la proporzionalità reale):
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
            "mode": "Opzione B (Analisi Semantica & IA)",
            "plagiarism_score": calculated_score,
            "score": calculated_score,
            "risk_level": risk,
            "summary_eval": result.get("summary_eval", "Analisi completata."),
            "critical_passages": critical
        }

    except Exception as parse_err:
        print(f"[LOG ERROR] Errore di decodifica JSON: {parse_err}")
        return {
            "mode": "Opzione B (Analisi Semantica & IA - Errore Formato)",
            "plagiarism_score": -1,
            "score": -1,
            "risk_level": "Errore",
            "summary_eval": "L'AI ha risposto ma con un formato invalido. Riprova la scansione.",
            "critical_passages": []
        }
