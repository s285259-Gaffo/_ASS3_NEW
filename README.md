# Assignment 3

## Istruzioni per l'esecuzione

Questo progetto utilizza un ambiente virtuale (virtual environment) per garantire che funzioni correttamente su qualsiasi PC, mantenendo le dipendenze isolate.

### Esecuzione su Windows
1. Aprire il terminale (PowerShell o Prompt dei comandi) in questa cartella.
2. (Facoltativo) Se la cartella `venv` non è già presente, crearla con:
   ```cmd
   python -m venv venv
   ```
3. Attivare il virtual environment:
   ```cmd
   .\venv\Scripts\activate
   ```
   *(Nota: se su PowerShell si riceve un errore di permessi, eseguire prima `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force`)*
4. Installare le dipendenze:
   ```cmd
   pip install -r requirements.txt
   ```
5. Eseguire il codice principale:
   ```cmd
   python runDMS.py
   ```

### Esecuzione su macOS / Linux
1. Aprire un terminale in questa cartella.
2. Creare il virtual environment:
   ```bash
   python3 -m venv venv
   ```
3. Attivarlo:
   ```bash
   source venv/bin/activate
   ```
4. Installare le dipendenze:
   ```bash
   pip install -r requirements.txt
   ```
5. Eseguire il codice principale:
   ```bash
   python3 runDMS.py
   ```