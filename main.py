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
    """Estrae il testo ed elimina Indice, Sommario e numeri di pagina per evitare falsi positivi."""
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

    # FILTRAGGIO ANTI-INDICE / SOMMARIO:
    lines = raw_text.split("\n")
    filtered_lines = []
    
    for line in lines:
        l = line.strip()
        if not l:
            continue
        
        # Ignora righe dell'indice (es. con sequenze di punti "..... 7" o parole chiave da sommario)
        if re.search(r'\.{2,}\s*\d+', l) or re.search(r'\.{4,}', l):
            continue
        if l.lower() in ["indice", "sommario", "table of contents", "executive summary"]:
            continue
            
        filtered_lines.append(l)

    cleaned_text = "\n".join(filtered_lines)
    return cleaned_text.strip()

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
# OPZIONE B: Rilevamento Plagio Semantico & Scrittura IA (Senza Falsi Positivi su Indici)
# ==========================================
@app.post("/analyze/ai")
async def analyze_option_b(file: UploadFile = File(...)):
    contents = await file.read()
    full_text = extract_clean_text(contents, file.filename.lower())
    
    if not full_text:
        raise HTTPException(status_code=400, detail="Impossibile estrarre testo valido dal file.")

    # Prendiamo i primi 4000 caratteri di TESTO DISCORSIVO (escludendo gli indici puliti in precedenza)
    text_sample = full_text[:4000].replace('"', "'").replace("\n", " ").replace("\\", "")

    prompt = f"""
    Sei uno strumento di analisi accademica per tesi di laurea.
    Analizza ESCLUSIVAMENTE IL TESTO DISCORSIVO sottostante:

    "{text_sample}"

    REGOLE TASSATIVE PER EVITARE FALSI POSITIVI:
    1. IGNORA STRUTTURE ED INDICI: Ignora del tutto eventuali titoli, numerazioni di capitoli o residui di sommari. NON considerarli mai come "testo generato da IA" o "plagio".
    2. VALUTA SOLO IL TESTO DISCORSIVO:
       - Riconosci il registro accademico umano (termini formali, trattazioni di casi studio come TRM Stampi S.r.l., contesti aziendali o teorici).
       - Penalizza SOLO ed ESCLUSIVAMENTE testo generico e vuoto tipico da ChatGPT (es: "Nel vasto panorama contemporaneo...", "È di fondamentale importanza notare che...") oppure definizioni di enciclopedia riprese parola per parola.

    CALCOLO DEL PUNTEGGIO REALE (0-100%):
    - 0-20%: Testo accademico reale, specifico, contestualizzato e ben articolato.
    - 21-50%: Stile leggermente generico ma autentico.
    - 51-100%: Testo palesemente sintetico, vuoto, generato da bot o copiato senza rielaborazione.

    Rispondi ESCLUSIVAMENTE con un oggetto JSON valido (nessun markdown, nessun testo di contorno):
    {{
        "plagiarism_score": <numero intero reale da 0 a 100>,
        "risk_level": "<Basso | Medio | Alto>",
        "summary_eval": "<Motivazione sintetica e contestualizzata>",
        "critical_passages": [
            {{
                "original_text": "<frase discorsiva realmente critica>",
                "type": "<Sospetto IA | Plagio Semantico>",
                "issue": "<spiegazione del perché la frase discorsiva è critica>",
                "rewritten_suggestion": "<suggerimento di riscrittura accademica>"
            }}
        ]
    }}
    """

    url = "https://text.pollinations.ai/"
    payload = {
        "messages": [
            {"role": "system", "content": "Sei un analista testuale accademico imparziale. Rispondi ESCLUSIVAMENTE con il JSON richiesto."},
            {"role": "user", "content": prompt}
        ],
        "model": "openai",
        "seed": 42
    }

    try:
        response = requests.post(url, json=payload, timeout=45)
        response.raise_for_status()
        response_text = response.text.strip()

        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}')
        
        if start_idx != -1 and end_idx != -1:
            clean_json = response_text[start_idx:end_idx+1]
        else:
            clean_json = response_text

        result = json.loads(clean_json)
        
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
            "summary_eval": f"Errore durante l'analisi AI: {str(e)[:60]}",
            "critical_passages": []
        }
