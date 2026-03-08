"""FastAPI router para DCRNN-lite de calidad del aire (multi-ciudad)."""

import json, math, datetime
from pathlib import Path
import numpy as np, joblib, torch
from fastapi import APIRouter, Query

ARTIFACTS = Path(__file__).parent / "artifacts"
router = APIRouter(prefix="/ml")

# ── Estaciones por ciudad ─────────────────────────────────────────────────────

CIUDADES_META = {
    "madrid":    {"nombre": "Madrid",    "center": [40.420, -3.703], "zoom": 11},
    "barcelona": {"nombre": "Barcelona", "center": [41.390,  2.154], "zoom": 12},
    "valencia":  {"nombre": "Valencia",  "center": [39.478, -0.366], "zoom": 12},
    "sevilla":   {"nombre": "Sevilla",   "center": [37.384, -5.991], "zoom": 12},
    "bilbao":    {"nombre": "Bilbao",    "center": [43.263, -2.935], "zoom": 13},
    "zaragoza":  {"nombre": "Zaragoza",  "center": [41.651, -0.889], "zoom": 13},
    "malaga":    {"nombre": "Málaga",    "center": [36.717, -4.417], "zoom": 13},
}

_STATIONS_CIUDAD = {
    "madrid": [
        {"id": "28079004", "nombre": "Pza. de España",        "lat": 40.4236, "lon": -3.7128, "codigo": 4},
        {"id": "28079008", "nombre": "Escuelas Aguirre",      "lat": 40.4220, "lon": -3.6819, "codigo": 8},
        {"id": "28079011", "nombre": "Avda. Ramón y Cajal",   "lat": 40.4516, "lon": -3.6773, "codigo": 11},
        {"id": "28079016", "nombre": "Arturo Soria",          "lat": 40.4484, "lon": -3.6393, "codigo": 16},
        {"id": "28079017", "nombre": "Villaverde",            "lat": 40.3473, "lon": -3.7136, "codigo": 17},
        {"id": "28079018", "nombre": "Farolillo",             "lat": 40.3960, "lon": -3.7244, "codigo": 18},
        {"id": "28079024", "nombre": "Casa de Campo",         "lat": 40.4205, "lon": -3.7477, "codigo": 24},
        {"id": "28079027", "nombre": "Barajas Pueblo",        "lat": 40.4743, "lon": -3.5800, "codigo": 27},
        {"id": "28079035", "nombre": "Pza. del Carmen",       "lat": 40.4186, "lon": -3.7034, "codigo": 35},
        {"id": "28079036", "nombre": "Moratalaz",             "lat": 40.4061, "lon": -3.6484, "codigo": 36},
        {"id": "28079038", "nombre": "Cuatro Caminos",        "lat": 40.4459, "lon": -3.7010, "codigo": 38},
        {"id": "28079039", "nombre": "Barrio del Pilar",      "lat": 40.4798, "lon": -3.7128, "codigo": 39},
        {"id": "28079040", "nombre": "Vallecas",              "lat": 40.3888, "lon": -3.6504, "codigo": 40},
        {"id": "28079047", "nombre": "Mendez Alvaro",         "lat": 40.3961, "lon": -3.6867, "codigo": 47},
        {"id": "28079048", "nombre": "Pza. Fernandez Ladreda","lat": 40.3894, "lon": -3.7239, "codigo": 48},
        {"id": "28079049", "nombre": "Pza. Castilla",         "lat": 40.4655, "lon": -3.6886, "codigo": 49},
        {"id": "28079050", "nombre": "Retiro",                "lat": 40.4087, "lon": -3.6824, "codigo": 50},
        {"id": "28079054", "nombre": "Ensanche Vallecas",     "lat": 40.3768, "lon": -3.6260, "codigo": 54},
        {"id": "28079055", "nombre": "Urb. Embajada",         "lat": 40.4747, "lon": -3.5885, "codigo": 55},
        {"id": "28079056", "nombre": "Pza. España (nuevo)",   "lat": 40.4234, "lon": -3.7124, "codigo": 56},
        {"id": "28079057", "nombre": "Sanchinarro",           "lat": 40.4970, "lon": -3.6524, "codigo": 57},
        {"id": "28079058", "nombre": "El Pardo",              "lat": 40.5180, "lon": -3.7744, "codigo": 58},
        {"id": "28079059", "nombre": "Juan Carlos I",         "lat": 40.4577, "lon": -3.6072, "codigo": 59},
        {"id": "28079060", "nombre": "Tres Olivos",           "lat": 40.5012, "lon": -3.6904, "codigo": 60},
    ],
    "barcelona": [
        {"id": "BCN001", "nombre": "Eixample",          "lat": 41.3897, "lon": 2.1540, "codigo": 1},
        {"id": "BCN002", "nombre": "Gràcia",             "lat": 41.4037, "lon": 2.1538, "codigo": 2},
        {"id": "BCN003", "nombre": "Sants",              "lat": 41.3742, "lon": 2.1341, "codigo": 3},
        {"id": "BCN004", "nombre": "Barceloneta",        "lat": 41.3797, "lon": 2.1876, "codigo": 4},
        {"id": "BCN005", "nombre": "Poblenou",           "lat": 41.3990, "lon": 2.1970, "codigo": 5},
        {"id": "BCN006", "nombre": "Horta",              "lat": 41.4320, "lon": 2.1600, "codigo": 6},
        {"id": "BCN007", "nombre": "Nou Barris",         "lat": 41.4400, "lon": 2.1760, "codigo": 7},
        {"id": "BCN008", "nombre": "Sant Andreu",        "lat": 41.4360, "lon": 2.1870, "codigo": 8},
        {"id": "BCN009", "nombre": "Les Corts",          "lat": 41.3840, "lon": 2.1320, "codigo": 9},
        {"id": "BCN010", "nombre": "Sarrià",             "lat": 41.4020, "lon": 2.1140, "codigo": 10},
    ],
    "valencia": [
        {"id": "VLC001", "nombre": "Ruzafa",             "lat": 39.4620, "lon": -0.3770, "codigo": 1},
        {"id": "VLC002", "nombre": "Cabanyal",           "lat": 39.4710, "lon": -0.3260, "codigo": 2},
        {"id": "VLC003", "nombre": "L'Eixample",         "lat": 39.4680, "lon": -0.3820, "codigo": 3},
        {"id": "VLC004", "nombre": "Benimaclet",         "lat": 39.4840, "lon": -0.3680, "codigo": 4},
        {"id": "VLC005", "nombre": "Campanar",           "lat": 39.4850, "lon": -0.4000, "codigo": 5},
        {"id": "VLC006", "nombre": "Patraix",            "lat": 39.4590, "lon": -0.3960, "codigo": 6},
        {"id": "VLC007", "nombre": "La Malva-rosa",      "lat": 39.4760, "lon": -0.3200, "codigo": 7},
        {"id": "VLC008", "nombre": "Quatre Carreres",    "lat": 39.4530, "lon": -0.3720, "codigo": 8},
    ],
    "sevilla": [
        {"id": "SEV001", "nombre": "Centro",             "lat": 37.3861, "lon": -5.9930, "codigo": 1},
        {"id": "SEV002", "nombre": "Bermejales",         "lat": 37.3720, "lon": -5.9900, "codigo": 2},
        {"id": "SEV003", "nombre": "Palmas Altas",       "lat": 37.3620, "lon": -5.9780, "codigo": 3},
        {"id": "SEV004", "nombre": "San Jerónimo",       "lat": 37.4120, "lon": -5.9840, "codigo": 4},
        {"id": "SEV005", "nombre": "Triana",             "lat": 37.3870, "lon": -6.0020, "codigo": 5},
        {"id": "SEV006", "nombre": "Nervión",            "lat": 37.3849, "lon": -5.9700, "codigo": 6},
    ],
    "bilbao": [
        {"id": "BIL001", "nombre": "Casco Viejo",        "lat": 43.2590, "lon": -2.9220, "codigo": 1},
        {"id": "BIL002", "nombre": "Deusto",             "lat": 43.2720, "lon": -2.9420, "codigo": 2},
        {"id": "BIL003", "nombre": "Indautxu",           "lat": 43.2638, "lon": -2.9365, "codigo": 3},
        {"id": "BIL004", "nombre": "Rekalde",            "lat": 43.2520, "lon": -2.9350, "codigo": 4},
        {"id": "BIL005", "nombre": "Basurto",            "lat": 43.2578, "lon": -2.9480, "codigo": 5},
    ],
    "zaragoza": [
        {"id": "ZGZ001", "nombre": "Centro",             "lat": 41.6561, "lon": -0.8773, "codigo": 1},
        {"id": "ZGZ002", "nombre": "Delicias",           "lat": 41.6432, "lon": -0.9010, "codigo": 2},
        {"id": "ZGZ003", "nombre": "Las Fuentes",        "lat": 41.6490, "lon": -0.8620, "codigo": 3},
        {"id": "ZGZ004", "nombre": "El Rabal",           "lat": 41.6680, "lon": -0.8780, "codigo": 4},
        {"id": "ZGZ005", "nombre": "Torrero",            "lat": 41.6353, "lon": -0.8810, "codigo": 5},
    ],
    "malaga": [
        {"id": "MLG001", "nombre": "Centro Histórico",   "lat": 36.7213, "lon": -4.4215, "codigo": 1},
        {"id": "MLG002", "nombre": "El Palo",            "lat": 36.7130, "lon": -4.3740, "codigo": 2},
        {"id": "MLG003", "nombre": "Churriana",          "lat": 36.6910, "lon": -4.5100, "codigo": 3},
        {"id": "MLG004", "nombre": "Soho",               "lat": 36.7162, "lon": -4.4270, "codigo": 4},
        {"id": "MLG005", "nombre": "Cruz de Humilladero","lat": 36.7120, "lon": -4.4400, "codigo": 5},
    ],
}

# ── Modelo (carga lazy, solo Madrid usa DCRNN real) ───────────────────────────

_model_cache = None
_scaler_cache = None
_adj_cache = None
_metadata_cache = {}


def _load_model():
    global _model_cache, _scaler_cache, _adj_cache, _metadata_cache
    if _model_cache is not None:
        return
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from train import DCRNN
    ckpt = torch.load(ARTIFACTS / "dcrnn_model.pt", map_location="cpu", weights_only=False)
    m = DCRNN(n_stations=ckpt["n_stations"], input_dim=ckpt["input_dim"],
              hidden_dim=ckpt["hidden_dim"], n_layers=ckpt["n_layers"],
              k_hops=ckpt["k_hops"], horizon=ckpt["horizon"])
    m.load_state_dict(ckpt["state_dict"])
    m.eval()
    _model_cache = (m, ckpt)
    _scaler_cache = joblib.load(ARTIFACTS / "scaler.joblib")
    _adj_cache = np.load(ARTIFACTS / "adj_matrix.npy")
    with open(ARTIFACTS / "metadata.json") as f:
        _metadata_cache = json.load(f)


def _ica_categoria(ica):
    if ica < 25:  return "Buena"
    if ica < 50:  return "Razonablemente buena"
    if ica < 75:  return "Regular"
    if ica < 100: return "Deficiente"
    if ica < 150: return "Muy deficiente"
    return "Extremadamente deficiente"


def _predict_madrid():
    """Usa el DCRNN entrenado para Madrid."""
    _load_model()
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from train import generate_synthetic_data, build_feature_matrix, STATIONS

    m, ckpt = _model_cache
    lookback, horizon = ckpt["lookback"], ckpt["horizon"]
    y_mean, y_std = ckpt["y_mean"], ckpt["y_std"]

    df = generate_synthetic_data(n_stations=len(STATIONS), n_hours=lookback + 1)
    station_map = {s["codigo"]: i for i, s in enumerate(STATIONS)}
    X_raw, y_ica, _ = build_feature_matrix(df, station_map)

    T, N, F = X_raw.shape
    X_norm = _scaler_cache.transform(X_raw.reshape(-1, F)).reshape(T, N, F).astype(np.float32)
    inp = torch.tensor(X_norm[-lookback:][np.newaxis], dtype=torch.float32)
    L_t = torch.tensor(_adj_cache, dtype=torch.float32)

    with torch.no_grad():
        pred_norm = m(inp, L_t)[0].numpy()

    pred_ica = np.clip(pred_norm * y_std + y_mean, 0, 500)
    stations = _STATIONS_CIUDAD["madrid"]
    results = []
    for i, st in enumerate(stations):
        ica_actual = float(y_ica[-1, i])
        ica_24h = [round(float(pred_ica[i, h]), 1) for h in range(horizon)]
        results.append({**st, "ica_actual": round(ica_actual, 1),
                        "ica_categoria": _ica_categoria(ica_actual),
                        "ica_prediccion_24h": ica_24h,
                        "ica_max_24h": round(max(ica_24h), 1),
                        "ica_max_categoria": _ica_categoria(max(ica_24h))})
    return results


def _predict_synthetic(ciudad: str):
    """Genera predicción sintética realista para ciudades sin modelo entrenado."""
    import math as m
    now = datetime.datetime.now()
    hour = now.hour
    stations = _STATIONS_CIUDAD[ciudad]

    # Parámetros base por ciudad (calidad del aire varía según tráfico y geografía)
    city_params = {
        "barcelona": {"base_no2": 28, "base_pm10": 22, "sea_factor": 0.85},
        "valencia":  {"base_no2": 22, "base_pm10": 18, "sea_factor": 0.75},
        "sevilla":   {"base_no2": 30, "base_pm10": 24, "sea_factor": 0.90},
        "bilbao":    {"base_no2": 20, "base_pm10": 16, "sea_factor": 0.80},
        "zaragoza":  {"base_no2": 26, "base_pm10": 21, "sea_factor": 1.10},
        "malaga":    {"base_no2": 18, "base_pm10": 15, "sea_factor": 0.70},
    }
    p = city_params.get(ciudad, {"base_no2": 25, "base_pm10": 20, "sea_factor": 0.9})

    import random
    rng = random.Random(42 + hash(ciudad) % 1000)

    results = []
    for i, st in enumerate(stations):
        # ICA actual basado en hora del día (punta 8h y 19h) con variación por estación
        hour_factor = (0.6 * m.exp(-0.5 * ((hour - 8) / 2.5) ** 2) +
                       0.4 * m.exp(-0.5 * ((hour - 19) / 2.5) ** 2) + 0.3)
        station_factor = 0.8 + 0.4 * rng.random()
        no2 = p["base_no2"] * hour_factor * station_factor * p["sea_factor"]
        pm10 = p["base_pm10"] * hour_factor * station_factor
        o3 = max(5, 45 - no2 * 0.6 + 10 * m.sin(2 * m.pi * (hour - 13) / 24))

        def no2_ica(v):
            bp = [(0,40,0,25),(40,90,25,50),(90,120,50,75),(120,230,75,100),(230,340,100,150),(340,999,150,300)]
            for lo_c,hi_c,lo_i,hi_i in bp:
                if lo_c<=v<hi_c: return lo_i+(v-lo_c)/(hi_c-lo_c)*(hi_i-lo_i)
            return 150
        def pm10_ica(v):
            bp = [(0,20,0,25),(20,35,25,50),(35,50,50,75),(50,100,75,100),(100,150,100,150),(150,999,150,300)]
            for lo_c,hi_c,lo_i,hi_i in bp:
                if lo_c<=v<hi_c: return lo_i+(v-lo_c)/(hi_c-lo_c)*(hi_i-lo_i)
            return 150
        def o3_ica(v):
            bp = [(0,60,0,25),(60,120,25,50),(120,180,50,75),(180,240,75,100),(240,380,100,150),(380,999,150,300)]
            for lo_c,hi_c,lo_i,hi_i in bp:
                if lo_c<=v<hi_c: return lo_i+(v-lo_c)/(hi_c-lo_c)*(hi_i-lo_i)
            return 150

        ica_actual = max(no2_ica(no2), pm10_ica(pm10), o3_ica(o3))

        # Predicción 24h
        ica_24h = []
        for h in range(24):
            hf = (0.6*m.exp(-0.5*((h-8)/2.5)**2)+0.4*m.exp(-0.5*((h-19)/2.5)**2)+0.3)
            n2 = p["base_no2"]*hf*station_factor*p["sea_factor"]
            p2 = p["base_pm10"]*hf*station_factor
            o2 = max(5, 45-n2*0.6+10*m.sin(2*m.pi*(h-13)/24))
            ica_24h.append(round(max(no2_ica(n2), pm10_ica(p2), o3_ica(o2)), 1))

        results.append({**st, "ica_actual": round(ica_actual, 1),
                        "ica_categoria": _ica_categoria(ica_actual),
                        "ica_prediccion_24h": ica_24h,
                        "ica_max_24h": round(max(ica_24h), 1),
                        "ica_max_categoria": _ica_categoria(max(ica_24h))})
    return results


def _predict(ciudad: str):
    if ciudad == "madrid":
        return _predict_madrid()
    return _predict_synthetic(ciudad)


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.get("/calidad-aire/ciudades")
def ciudades():
    return {"ciudades": [{"id": cid, **meta} for cid, meta in CIUDADES_META.items()]}

@router.get("/calidad-aire/prediccion")
def prediccion(ciudad: str = Query("madrid")):
    try:
        stations = _predict(ciudad)
        now = datetime.datetime.now()
        horas = [(now + datetime.timedelta(hours=h+1)).strftime("%H:%M") for h in range(24)]
        meta = CIUDADES_META.get(ciudad, {})
        return {"ok": True, "timestamp": now.isoformat(), "ciudad": ciudad,
                "nombre_ciudad": meta.get("nombre", ciudad),
                "center": meta.get("center"), "zoom": meta.get("zoom"),
                "horas_prediccion": horas, "estaciones": stations}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@router.get("/calidad-aire/estaciones")
def estaciones(ciudad: str = Query("madrid")):
    try:
        return {"ok": True, "ciudad": ciudad, "estaciones": _predict(ciudad)}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@router.get("/calidad-aire/stats")
def stats():
    try:
        _load_model()
        return {"ok": True, **_metadata_cache}
    except Exception as e:
        return {"ok": False, "error": str(e)}
