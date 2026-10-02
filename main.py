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
# OPZIONE B: Analisi Semantica Completa (Plagio IA, Parafrasi & Stile)
# ==========================================
@app.post("/analyze/ai")
async def analyze_option_b(file: UploadFile = File(...)):
    contents = await file.read()
    full_text = extract_text(contents, file.filename.lower())
    
    text_sample = full_text[:4000]

    prompt = f"""
    Sei un docente universitario e revisore accademico di massima esperienza.
    Analizza il seguente estratto di una tesi di laurea:

    \"\"\"
    {text_sample}
    \"\"\"

    ISTRUZIONI DI ANALISI RIGOROSE:
    1. IGNORA STRUTTURA E FORMA: Non considerare mai errori o rischi la presenza di titoli di capitoli, numerazioni (es. 1.1, Capitolo 2), note bibliografiche o indice. È la normale struttura di una tesi.
    2. RILEVAZIONE PLAGIO IA: Cerca costrutti artificiali tipici dei modelli LLM (ChatGPT/Claude), come connettivi meccanici ("È importante sottolineare che", "In sintesi"), tono eccessivamente neutro o privo di analisi critica.
    3. RILEVAZIONE PARAFRASI/PLAGIO SEMANTICO: Identifica periodi che sembrano tradotti o rielaborati superficialmente per mascherare una fonte originale.
    4. REVISIONE SINTATTICO-STILISTICA: Individua frasi con sintassi debole, ripetizioni o registro non adeguatamente accademico.
    5. SELEZIONA OBBLIGATORIAMENTE da 2 a 5 passaggi critici e proponi per ciascuno una riscrittura accademica formale.

    Rispondi ESCLUSIVAMENTE con un JSON che rispetti questo formato esatto:
    {{
        "plagiarism_score": 20,
        "ai_generated_probability": 15,
        "risk_level": "Basso",
        "summary_eval": "Valutazione sintetica complessiva su stile, originalità e potenziale uso di IA...",
        "critical_passages": [
            {{
                "original_text": "citazione esatta del passaggio dalla tesi",
                "type": "Sospetto IA",
                "issue": "spiegazione del perché il passaggio sembra generato da IA, parafrasato o debole stilisticamente",
                "rewritten_suggestion": "proposta di riscrittura rigorosa in perfetto stile accademico"
            }}
        ]
    }}
    Nota per il campo 'type': usa solo una tra queste diciture: 'Sospetto IA', 'Parafrasi Superficiale', 'Debolezza Sintattica' o 'Stile da Migliorare'.
    """

    url = "https://text.pollinations.ai/"
    payload = {
        "messages": [
            {
                "role": "system", 
                "content": "Sei un revisore accademico. Rispondi SEMPRE ed ESCLUSIVAMENTE con un JSON valido fornendo score, analisi IA e suggerimenti di riscrittura."
            },
            {"role": "user", "content": prompt}
        ],
        "model": "llama",
        "jsonMode": True
    }

    try:
        response = requests.post(url, json=payload, timeout=60)
        response_text = response.text.strip()

        # Pulizia rigida del JSON da eventuale sintassi markdown
        clean_json = re.sub(r"^```json\s*", "", response_text, flags=re.MULTILINE)
        clean_json = re.sub(r"^```\s*", "", clean_json, flags=re.MULTILINE)
        clean_json = re.sub(r"```$", "", clean_json, flags=re.MULTILINE).strip()

        result = json.loads(clean_json)
        
        # Estrazione sicura del punteggio per il frontend
        score_val = result.get("plagiarism_score", result.get("score", result.get("ai_generated_probability", 0)))
        
        result["plagiarism_score"] = score_val
        result["score"] = score_val
        result["mode"] = "Opzione B (Analisi Semantica, IA & Stile - Llama)"
        
        if "critical_passages" not in result or not isinstance(result["critical_passages"], list):
            result["critical_passages"] = []

        return result

    except Exception as e:
        return {
            "mode": "Opzione B (Analisi Semantica, IA & Stile - Llama)",
            "plagiarism_score": 0,
            "score": 0,
            "risk_level": "Basso",
            "summary_eval": "Errore durante l'elaborazione dell'analisi semantica.",
            "critical_passages": []
        }
