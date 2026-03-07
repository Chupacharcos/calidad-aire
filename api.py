"""
Servidor FastAPI standalone para el proyecto calidad-aire.
Puerto: 8091

Arrancar:
  cd /var/www/calidad-aire
  venv/bin/uvicorn api:app --host 0.0.0.0 --port 8091
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="ML Calidad del Aire — DCRNN-lite",
    version="1.0",
    description="Predicción de ICA (Índice de Calidad del Aire) a 24h mediante DCRNN sobre grafo de estaciones.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

from router import router
app.include_router(router)


@app.get("/")
def root():
    return {"service": "ml-calidad-aire", "status": "ok", "port": 8091}
