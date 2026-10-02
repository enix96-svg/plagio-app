<!DOCTYPE html>
<html lang="it">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Test Box Antiplagio</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-slate-900 text-slate-100 p-6 font-sans">
    <div class="max-w-md mx-auto bg-slate-800 p-6 rounded-2xl border border-slate-700 space-y-4">
        <h2 class="text-lg font-bold text-indigo-400">🧪 Box di Test Integrato</h2>
        <p class="text-xs text-slate-400">Verifica l'estrazione testo e la connessione API prima di andare in produzione.</p>
        
        <div>
            <label class="block text-xs font-semibold mb-1">1. Inserisci un testo di prova:</label>
            <textarea id="testText" rows="3" class="w-full bg-slate-900 border border-slate-700 rounded-xl p-2 text-xs text-slate-200" placeholder="Incolla un paragrafo della tesi qui..."></textarea>
        </div>

        <div>
            <label class="block text-xs font-semibold mb-1">2. Endpoint Backend (Render/Python):</label>
            <input type="text" id="endpointUrl" class="w-full bg-slate-900 border border-slate-700 rounded-xl p-2 text-xs text-slate-200" value="http://localhost:8000/analyze/search">
        </div>

        <button onclick="runTest()" id="btnTest" class="w-full bg-indigo-600 hover:bg-indigo-500 text-white font-bold py-2 rounded-xl text-xs">
            Esegui Test di Connessione
        </button>

        <div id="testOutput" class="hidden text-xs bg-slate-900 p-3 rounded-xl border border-slate-700 font-mono text-emerald-400 whitespace-pre-wrap"></div>
    </div>

    <script>
        async function runTest() {
            const text = document.getElementById('testText').value;
            const url = document.getElementById('endpointUrl').value;
            const output = document.getElementById('testOutput');
            const btn = document.getElementById('btnTest');

            if (!text) { alert("Inserisci prima del testo nel box!"); return; }

            btn.disabled = true;
            btn.innerText = "Invio in corso...";
            output.classList.add('hidden');

            try {
                // Simula l'invio come file per testare la compatibilità dell'API
                const blob = new Blob([text], { type: 'text/plain' });
                const formData = new FormData();
                formData.append("file", blob, "test_doc.docx");

                const res = await fetch(url, { method: 'POST', body: formData });
                const data = await res.json();
                
                output.innerText = JSON.stringify(data, null, 2);
                output.classList.remove('hidden');
            } catch (err) {
                output.innerText = "❌ Errore di connessione:\n" + err.message;
                output.classList.remove('hidden');
                output.className = output.className.replace('text-emerald-400', 'text-rose-400');
            } finally {
                btn.disabled = false;
                btn.innerText = "Esegui Test di Connessione";
            }
        }
    </script>
</body>
</html>
