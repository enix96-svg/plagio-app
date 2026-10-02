import os
import io
import json
import docx
import pypdf
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from duckduckgo_search import DDGS
from google import genai

app = FastAPI(title="Real Anti-Plagiarism API")

# Configurazione CORS per comunicare con Netlify
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

def extract_text(file_bytes: bytes, filename: str) -> str:
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

@app.get("/")
def read_root():
    return {"status": "online", "message": "Backend Antiplagio Attivo"}

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

@app.post("/analyze/ai")
async def analyze_option_b(file: UploadFile = File(...)):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Chiave GEMINI_API_KEY non configurata su Render.")

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
    """

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        # Prova prima con gemini-2.5-flash, in caso di errore passa a gemini-1.5-flash
        models_to_try = ['gemini-2.5-flash', 'gemini-1.5-flash']
        response = None
        last_error = None

        for model_name in models_to_try:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                )
                if response:
                    break
            except Exception as e:
                last_error = e
                continue

        if not response:
            raise HTTPException(status_code=500, detail=f"Errore chiamate modelli AI: {str(last_error)}")

        clean_text = response.text.strip()
        if clean_text.startswith("```json"):
            clean_text = clean_text[7:]
        if clean_text.startswith("```"):
            clean_text = clean_text[3:]
        if clean_text.endswith("```"):
            clean_text = clean_text[:-3]

        result = json.loads(clean_text.strip())
        result["mode"] = "Opzione B (Analisi Semantica AI)"
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore analisi AI: {str(e)}")
