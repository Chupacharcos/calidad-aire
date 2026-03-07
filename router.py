"""
FastAPI router para el modelo DCRNN-lite de calidad del aire.
Endpoints:
  GET /ml/calidad-aire/prediccion   → ICA predicho para las próximas 24h por estación
  GET /ml/calidad-aire/estaciones   → lista de estaciones con ICA actual
  GET /ml/calidad-aire/stats        → métricas del modelo
"""

import json
import math
import datetime
from pathlib import Path
from functools import lru_cache
from typing import Any

import numpy as np
import joblib
import torch
from fastapi import APIRouter

ARTIFACTS = Path(__file__).parent / "artifacts"

router = APIRouter(prefix="/ml")


# ── Carga lazy del modelo ──────────────────────────────────────────────────────

_model = None
_scaler = None
_adj = None
_stations: list[dict] = []
_metadata: dict = {}


def _load_artifacts():
    global _model, _scaler, _adj, _stations, _metadata

    if _model is not None:
        return

    # Importar clases del modelo desde train.py
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from train import DCRNN, STATIONS, compute_ica

    checkpoint = torch.load(ARTIFACTS / "dcrnn_model.pt", map_location="cpu", weights_only=False)
    m = DCRNN(
        n_stations=checkpoint["n_stations"],
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        n_layers=checkpoint["n_layers"],
        k_hops=checkpoint["k_hops"],
        horizon=checkpoint["horizon"],
    )
    m.load_state_dict(checkpoint["state_dict"])
    m.eval()
    _model = (m, checkpoint)

    _scaler = joblib.load(ARTIFACTS / "scaler.joblib")
    _adj = np.load(ARTIFACTS / "adj_matrix.npy")

    with open(ARTIFACTS / "stations.json", encoding="utf-8") as f:
        _stations = json.load(f)

    with open(ARTIFACTS / "metadata.json") as f:
        _metadata = json.load(f)


def _predict_current() -> list[dict]:
    """Genera predicción ICA 24h usando datos sintéticos del momento actual."""
    _load_artifacts()
    from train import generate_synthetic_data, build_feature_matrix, compute_ica, STATIONS

    m, ckpt = _model
    lookback = ckpt["lookback"]
    horizon = ckpt["horizon"]
    y_mean = ckpt["y_mean"]
    y_std = ckpt["y_std"]

    # Generar ventana de datos recientes (simulada)
    df = generate_synthetic_data(n_stations=len(STATIONS), n_hours=lookback + 1)
    station_map = {s["codigo"]: i for i, s in enumerate(STATIONS)}
    X_raw, y_ica, _ = build_feature_matrix(df, station_map)

    T, N, F = X_raw.shape
    X_flat = X_raw.reshape(-1, F)
    X_norm = _scaler.transform(X_flat).reshape(T, N, F).astype(np.float32)

    # Tomar últimos LOOKBACK pasos
    window = X_norm[-lookback:]  # [lookback, N, F]
    inp = torch.tensor(window[np.newaxis], dtype=torch.float32)  # [1, lookback, N, F]
    L_tensor = torch.tensor(_adj, dtype=torch.float32)

    with torch.no_grad():
        pred_norm = m(inp, L_tensor)  # [1, N, horizon]

    pred_ica = pred_norm[0].numpy() * y_std + y_mean  # [N, horizon]
    pred_ica = np.clip(pred_ica, 0, 500)

    # ICA actual (última hora de datos sintéticos, ajustado por hora real)
    now = datetime.datetime.now()
    hour = now.hour

    results = []
    for i, st in enumerate(STATIONS):
        ica_actual = float(y_ica[-1, i])
        ica_24h = [round(float(pred_ica[i, h]), 1) for h in range(horizon)]
        ica_max_24h = float(max(ica_24h))

        results.append({
            "id": st["id"],
            "nombre": st["nombre"],
            "lat": st["lat"],
            "lon": st["lon"],
            "codigo": st["codigo"],
            "ica_actual": round(ica_actual, 1),
            "ica_categoria": _ica_categoria(ica_actual),
            "ica_prediccion_24h": ica_24h,
            "ica_max_24h": round(ica_max_24h, 1),
            "ica_max_categoria": _ica_categoria(ica_max_24h),
        })

    return results


def _ica_categoria(ica: float) -> str:
    if ica < 25:
        return "Buena"
    if ica < 50:
        return "Razonablemente buena"
    if ica < 75:
        return "Regular"
    if ica < 100:
        return "Deficiente"
    if ica < 150:
        return "Muy deficiente"
    return "Extremadamente deficiente"


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.get("/calidad-aire/prediccion")
def prediccion():
    """ICA predicho para las próximas 24h por estación (DCRNN-lite)."""
    try:
        stations = _predict_current()
        now = datetime.datetime.now()
        horas = [(now + datetime.timedelta(hours=h + 1)).strftime("%H:%M") for h in range(24)]
        return {
            "ok": True,
            "timestamp": now.isoformat(),
            "horas_prediccion": horas,
            "estaciones": stations,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.get("/calidad-aire/estaciones")
def estaciones():
    """Lista de estaciones con ICA actual."""
    try:
        stations = _predict_current()
        return {"ok": True, "estaciones": stations}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.get("/calidad-aire/stats")
def stats():
    """Métricas del modelo entrenado."""
    try:
        _load_artifacts()
        return {"ok": True, **_metadata}
    except Exception as e:
        return {"ok": False, "error": str(e)}
