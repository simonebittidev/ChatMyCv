FROM node:20-alpine AS frontend-builder
WORKDIR /app/client
COPY client/ ./
RUN npm install && npm run build

# Usa Python 3.11 slim per immagini più leggere
FROM python:3.11-slim
WORKDIR /app

# Copia il backend e requirements
WORKDIR /app
COPY . /app
# COPY backend/ ./backend/

# Copia il frontend già esportato nella posizione che il backend si aspetta
COPY --from=frontend-builder /app/client/out ./client/out

# Installa le dipendenze
RUN pip install --no-cache-dir -r requirements.txt

# Esponi la porta 80
EXPOSE 80

# Comando di avvio (adatta se usi FastAPI con Uvicorn)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5005"]