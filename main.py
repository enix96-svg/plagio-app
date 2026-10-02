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

# Configurazione CORS per la comunicazione sicura con Netlify
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
# OPZIONE B: Analisi Semantica & Rilevamento IA (Prompt Accademico)
# ==========================================
@app.post("/analyze/ai")
async def analyze_option_b(file: UploadFile = File(...)):
    contents = await file.read()
    full_text = extract_text(contents, file.filename.lower())
    
    # Campione di testo per l'analisi approfondita
    text_sample = full_text[:4000]

    prompt = f"""
    Sei un professore universitario e membro di una commissione di laurea con decennale esperienza nell'analisi e nella revisione di tesi e paper accademici.

    CONTESTO ED ESIGENZE DI VALUTAZIONE:
    Il testo che segue è un estratto formale da una TESI DI LAUREA. 
    1. È del tutto NORMALE e CORRETTO che contenga numerazioni di capitoli/paragrafi (es. "Capitolo 1", "1.1"), citazioni bibliografiche, note a piè di pagina o formalismi accademici. NON segnalare queste strutture standard come errori o problemi.
    2. Concentrati ESCLUSIVAMENTE sul contenuto stilistico, la paternità del testo e la qualità della scrittura.

    IL TUO COMPITO:
    Analizza l'estratto fornito per identificare:
    - Rischio Generazione IA: fraseggi stereotipati tipici dei modelli LLM (es. "In conclusione...", "È fondamentale sottolineare che...", transizioni artificiali, mancanza di voce critica personale).
    - Rischio Parafrasi Sospetta / Plagio Semantico: periodi che sembrano tradotti letteralmente da fonti estere o parafrasati in modo superficiale per mascherare il testo originale.
    - Qualità e Fluidità Accademica: passaggi confusi o sintatticamente deboli che richiedono una riscrittura formale.

    TESTO DELLA TESI DA ANALIZZARE:
    \"\"\"
    {text_sample}
    \"\"\"

    REGOLE RIGIDE SULLA RISPOSTA:
    Rispondi TASSATIVAMENTE ed ESCLUSIVAMENTE in formato JSON valido (senza blocchi di codice markdown, senza spiegazioni aggiuntive fuori dal JSON).

    Struttura JSON richiesta:
    {{
        "plagiarism_score": <numero intero da 0 a 100 indicante la percentuale stimata di rischio totale plagio/IA>,
        "ai_generated_probability": <numero intero da 0 a 100 indicante la probabilità che il testo sia generato da IA>,
        "risk_level": "<Basso | Medio | Alto>",
        "summary_eval": "<valutazione sintetica complessiva della tesi in 2-3 frasi, tono formale e professionale>",
        "critical_passages": [
            {{
                "original_text": "<citazione esatta della frase o del passaggio sospetto dalla tesi>",
                "type": "<Sospetto IA | Parafrasi Superficiale | Sintassi Debole>",
                "issue": "<spiegazione accademica precisa del perché il passaggio è critico>",
                "rewritten_suggestion": "<proposta di riscrittura rigorosa e ad alto livello accademico>"
            }}
        ]
    }}
    """

    url = "https://text.pollinations.ai/"
    payload = {
        "messages": [
            {
                "role": "system", 
                "content": "Sei un revisore accademico esperto. Rispondi SEMPRE ed ESCLUSIVAMENTE con un oggetto JSON valido, senza formattazione Markdown o testo introduttivo."
            },
            {"role": "user", "content": prompt}
        ],
        "jsonMode": True
    }

    try:
        response = requests.post(url, json=payload, timeout=60)
        response_text = response.text.strip()

        # Pulizia rigorosa del JSON da eventuale sintassi markdown residua
        clean_json = re.sub(r"^```json\s*", "", response_text, flags=re.MULTILINE)
        clean_json = re.sub(r"^```\s*", "", clean_json, flags=re.MULTILINE)
        clean_json = re.sub(r"```$", "", clean_json, flags=re.MULTILINE).strip()

        result = json.loads(clean_json)
        result["mode"] = "Opzione B (Analisi Semantica & Rilevamento IA)"
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore durante l'analisi dell'estratto: {str(e)}")
