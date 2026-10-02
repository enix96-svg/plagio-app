# ==========================================
# OPZIONE B: Analisi Semantica Completa (Valutazione Reale)
# ==========================================
@app.post("/analyze/ai")
async def analyze_option_b(file: UploadFile = File(...)):
    contents = await file.read()
    full_text = extract_text(contents, file.filename.lower())
    
    if not full_text:
        raise HTTPException(status_code=400, detail="Impossibile estrarre testo dal file.")

    # Pulizia profonda per evitare che i caratteri speciali rompano la richiesta JSON
    text_sample = full_text[:4000].replace('"', "'").replace("\n", " ").replace("\\", "")

    # Prompt estremamente esplicito: forniamo un ESEMPIO con numeri reali, non con i simboli < >
    prompt = f"""Analizza questo testo e valuta la probabilità che sia generato da IA o plagiato.
    Testo: "{text_sample}"

    REGOLE DI VALUTAZIONE (Sii severo):
    1. Se noti uno stile robotico, frasi fatte tipiche delle IA (es. "Nel vasto panorama", "È fondamentale ricordare", "In sintesi") o testo puramente enciclopedico, il punteggio DEVE essere tra 75 e 100.
    2. Se il testo è originale, con argomentazioni personali e fonti citate correttamente, il punteggio deve essere tra 0 e 25.

    Devi rispondere SOLO ed ESCLUSIVAMENTE con un oggetto JSON valido. Nessun testo introduttivo o conclusivo.
    Usa ESATTAMENTE questo formato (assicurati che plagiarism_score sia un NUMERO intero, non una stringa):
    {{
        "plagiarism_score": 85,
        "risk_level": "Alto",
        "summary_eval": "Motivazione sintetica del punteggio assegnato...",
        "critical_passages": [
            {{
                "original_text": "Inserisci qui una frase esatta presa dal testo che risulta sospetta",
                "type": "Sospetto IA",
                "issue": "Spiega perché questa frase sembra generata da IA o plagiata",
                "rewritten_suggestion": "Scrivi una proposta di miglioramento in stile accademico"
            }}
        ]
    }}
    """

    url = "https://text.pollinations.ai/"
    payload = {
        "messages": [
            {"role": "system", "content": "Sei un'API che restituisce ESCLUSIVAMENTE codice JSON valido. Non usare formattazione markdown (```json). Restituisci solo l'oggetto tra parentesi graffe."},
            {"role": "user", "content": prompt}
        ],
        "model": "openai",
        "jsonMode": True, # Forza il modello a validare il JSON (se supportato)
        "seed": 42
    }

    try:
        response = requests.post(url, json=payload, timeout=45)
        response.raise_for_status() # Lancia errore se il server remoto non risponde con 200 OK
        response_text = response.text.strip()

        # Estrazione JSON infallibile (cerca la prima { e l'ultima })
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}')
        
        if start_idx != -1 and end_idx != -1:
            clean_json = response_text[start_idx:end_idx+1]
        else:
            clean_json = response_text

        result = json.loads(clean_json)
        
        # Ci assicuriamo che lo score sia un intero valido, anche se l'AI dovesse restituire una stringa
        raw_score = result.get("plagiarism_score", 0)
        try:
            score_val = int(raw_score)
        except ValueError:
            score_val = 50 # Se l'AI ha scritto testo, assegniamo un 50% di default per sicurezza
        
        result["plagiarism_score"] = score_val
        result["score"] = score_val
        result["mode"] = "Opzione B (Analisi Semantica & IA)"
        
        if "critical_passages" not in result or not isinstance(result["critical_passages"], list):
            result["critical_passages"] = []

        return result

    except Exception as e:
        print(f"Errore chiamata AI: {e}")
        # In caso di errore API non diamo più 0 (che illude l'utente), ma restituiamo un errore gestibile
        return {
            "mode": "Opzione B (Analisi Semantica & IA - Errore API)",
            "plagiarism_score": -1, 
            "score": -1,
            "risk_level": "Errore",
            "summary_eval": f"Errore di connessione con il motore IA gratuito. Dettaglio: {str(e)[:50]}...",
            "critical_passages": []
        }
