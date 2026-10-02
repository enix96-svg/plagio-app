import os
import io
import docx
import pypdf
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from duckduckgo_search import DDGS
import google.generativeai as genai

app = FastAPI(title="Real Anti-Plagiarism API")

# Abilita CORS per permettere le chiamate dal frontend Netlify
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configura l'API Key per l'Opzione B (Gemini AI - Gratuita su Google AI Studio)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "INSERISCI_LA_TUA_GEMINI_KEY_QUI")
if GEMINI_API_KEY != "INSERISCI_LA_TUA_GEMINI_KEY_QUI":
    genai.configure(api_key=GEMINI_API_KEY)

def extract_text(file_bytes: bytes, filename: str) -> str:
    """Estrae il testo puro da file .docx o .pdf"""
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

# ==========================================
# OPZIONE A: Ricerca Reale sul Web (DuckDuckGo)
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
# OPZIONE B: Analisi Semantica e Parafrasi con AI
# ==========================================
@app.post("/analyze/ai")
async def analyze_option_b(file: UploadFile = File(...)):
    contents = await file.read()
    full_text = extract_text(contents, file.filename.lower())
    
    text_sample = full_text[:8000]

    prompt = f"""
    Sei un revisore accademico esperto in anti-plagio e stile di tesi universitarie.
    Analizza il seguente estratto di tesi:

    "{text_sample}"

    Fornisci una risposta JSON valida con la seguente struttura:
    {{
        "plagiarism_score": <numero da 0 a 100>,
        "risk_level": "<Basso | Medio | Alto>",
        "critical_passages": [
            {{
                "original_text": "<frase sospetta o mal citata>",
                "issue": "<motivo per cui è a rischio o sembra parafrasata>",
                "rewritten_suggestion": "<versione riformulata in perfetto stile accademico>"
            }}
        ]
    }}
    Rispondi SOLO con il JSON valido.
    """

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        clean_json = response.text.replace("```json", "").replace("```", "").strip()
        import json
        result = json.loads(clean_json)
        result["mode"] = "Opzione B (Analisi Semantica AI)"
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore analisi AI: {str(e)}")
