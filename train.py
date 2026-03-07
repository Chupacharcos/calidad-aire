"""
Entrenamiento del modelo DCRNN-lite para predicción de calidad del aire en Madrid.

Datos:
  - Red de Vigilancia de la Calidad del Aire de Madrid (datos.madrid.es)
  - Meteorología: Open-Meteo API (sin API key)

Arquitectura:
  - GRU con Diffusion Graph Convolution (K=2 hops, puro PyTorch)
  - Entrada: últimas 12h por estación (NO2, PM10, O3, wind_speed, wind_sin, wind_cos, temp, hour_sin, hour_cos)
  - Salida: ICA predicho para las próximas 24h por estación

Ejecutar una vez para generar artifacts/:
  python3 train.py
"""

import json
import math
import io
import warnings
import zipfile
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import joblib
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

ARTIFACTS = Path(__file__).parent / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

# ── Configuración ─────────────────────────────────────────────────────────────

LOOKBACK   = 12   # horas de historia como entrada
HORIZON    = 24   # horas de predicción
BATCH_SIZE = 32
EPOCHS     = 40
LR         = 1e-3
HIDDEN_DIM = 32
N_LAYERS   = 2
K_HOPS     = 2    # pasos de difusión en el grafo
DIST_SIGMA = 0.05  # sigma del kernel gaussiano (grados lat/lon ~5km)
DIST_THRESHOLD = 0.15  # umbral máximo de distancia para conexión (~15km)

# Contaminantes usados (los que tienen mejor cobertura en Madrid)
POLLUTANTS = ["NO2", "PM10", "O3"]

# Features temporales + meteorológicas (sin contaminantes, se añaden después)
META_FEATURES = ["wind_speed", "wind_sin", "wind_cos", "temp", "hour_sin", "hour_cos"]

# Estaciones de Madrid con coordenadas (Red de Vigilancia)
# Fuente: datos.madrid.es / calidad del aire
STATIONS = [
    {"id": "28079004", "nombre": "Pza. de España",         "lat": 40.4236, "lon": -3.7128, "codigo": 4},
    {"id": "28079008", "nombre": "Escuelas Aguirre",       "lat": 40.4220, "lon": -3.6819, "codigo": 8},
    {"id": "28079011", "nombre": "Avda. Ramón y Cajal",    "lat": 40.4516, "lon": -3.6773, "codigo": 11},
    {"id": "28079016", "nombre": "Arturo Soria",           "lat": 40.4484, "lon": -3.6393, "codigo": 16},
    {"id": "28079017", "nombre": "Villaverde",             "lat": 40.3473, "lon": -3.7136, "codigo": 17},
    {"id": "28079018", "nombre": "Farolillo",              "lat": 40.3960, "lon": -3.7244, "codigo": 18},
    {"id": "28079024", "nombre": "Casa de Campo",          "lat": 40.4205, "lon": -3.7477, "codigo": 24},
    {"id": "28079027", "nombre": "Barajas Pueblo",         "lat": 40.4743, "lon": -3.5800, "codigo": 27},
    {"id": "28079035", "nombre": "Pza. del Carmen",        "lat": 40.4186, "lon": -3.7034, "codigo": 35},
    {"id": "28079036", "nombre": "Moratalaz",              "lat": 40.4061, "lon": -3.6484, "codigo": 36},
    {"id": "28079038", "nombre": "Cuatro Caminos",         "lat": 40.4459, "lon": -3.7010, "codigo": 38},
    {"id": "28079039", "nombre": "Barrio del Pilar",       "lat": 40.4798, "lon": -3.7128, "codigo": 39},
    {"id": "28079040", "nombre": "Vallecas",               "lat": 40.3888, "lon": -3.6504, "codigo": 40},
    {"id": "28079047", "nombre": "Mendez Alvaro",          "lat": 40.3961, "lon": -3.6867, "codigo": 47},
    {"id": "28079048", "nombre": "Pza. Fernandez Ladreda", "lat": 40.3894, "lon": -3.7239, "codigo": 48},
    {"id": "28079049", "nombre": "Pza. Castilla",          "lat": 40.4655, "lon": -3.6886, "codigo": 49},
    {"id": "28079050", "nombre": "Retiro",                 "lat": 40.4087, "lon": -3.6824, "codigo": 50},
    {"id": "28079054", "nombre": "Ensanche Vallecas",      "lat": 40.3768, "lon": -3.6260, "codigo": 54},
    {"id": "28079055", "nombre": "Urb. Embajada",         "lat": 40.4747, "lon": -3.5885, "codigo": 55},
    {"id": "28079056", "nombre": "Pza. España (nuevo)",   "lat": 40.4234, "lon": -3.7124, "codigo": 56},
    {"id": "28079057", "nombre": "Sanchinarro",            "lat": 40.4970, "lon": -3.6524, "codigo": 57},
    {"id": "28079058", "nombre": "El Pardo",               "lat": 40.5180, "lon": -3.7744, "codigo": 58},
    {"id": "28079059", "nombre": "Juan Carlos I",          "lat": 40.4577, "lon": -3.6072, "codigo": 59},
    {"id": "28079060", "nombre": "Tres Olivos",            "lat": 40.5012, "lon": -3.6904, "codigo": 60},
]

# Códigos de magnitud del Ayuntamiento de Madrid
MAGNITUD_CODES = {
    "SO2":  1,
    "CO":   6,
    "NO":   7,
    "NO2":  8,
    "PM2.5": 9,
    "PM10": 10,
    "NOx":  12,
    "O3":   14,
    "TOL":  20,
    "BEN":  30,
    "EBE":  35,
    "TCH":  42,
    "CH4":  43,
    "NMHC": 44,
}


# ── Descarga de datos ─────────────────────────────────────────────────────────

def download_madrid_air_quality(year: int) -> pd.DataFrame:
    """
    Descarga datos horarios de calidad del aire de Madrid para un año dado.
    Fuente: Portal de Datos Abiertos del Ayuntamiento de Madrid.
    Formato: CSV anual con columnas ESTACION, MAGNITUD, Hxx (hora 01..24).
    """
    print(f"  Descargando datos Madrid {year}...")
    url = (
        f"https://datos.madrid.es/egob/catalogo/212531-{year - 2000 + 7}-aire-datos-horarios.zip"
        if year >= 2019
        else f"https://datos.madrid.es/egob/catalogo/212531-{year - 2000 + 7}-aire-datos-horarios.zip"
    )
    # URL patrón real del portal de Madrid
    urls_to_try = [
        f"https://datos.madrid.es/egob/catalogo/212531-{year - 2001}-calidad-aire-horario.zip",
        f"https://datos.madrid.es/egob/catalogo/212531-0-calidad-aire-horario.zip",
    ]

    # Intentamos con la URL directa del CSV (datos abiertos Madrid)
    # Los datos están en formato horario con columnas para cada hora del día
    base_url = "https://datos.madrid.es/egob/catalogo/"

    # Mapeo de años a IDs de dataset conocidos
    dataset_ids = {
        2023: "212531-7749",
        2022: "212531-7748",
        2021: "212531-7747",
        2020: "212531-7746",
        2019: "212531-7745",
    }

    dfs = []
    for month in range(1, 13):
        month_str = f"{month:02d}"
        year_str = str(year)
        csv_url = (
            f"https://datos.madrid.es/egob/catalogo/212531-7749-calidad-aire-horario.zip"
        )
        break  # usaremos el CSV anual abajo

    # URL real del CSV anual
    csv_url = f"https://datos.madrid.es/egob/catalogo/212531-{year}-calidad-aire-horario.zip"

    # Intentar descarga directa del CSV via la API de datos abiertos
    api_url = (
        "https://datos.madrid.es/egob/catalogo/212531-7749-calidad-aire-horario.zip"
        if year == 2023
        else f"https://datos.madrid.es/egob/catalogo/212531-{year - 2001 + 1}-calidad-aire-horario.zip"
    )

    return None  # Se manejará en download_all_data con fallback


def download_madrid_csv_direct(year: int) -> pd.DataFrame | None:
    """Descarga CSV de calidad del aire de Madrid directamente."""
    # La URL real del portal de Madrid para datos horarios
    # Formato: anual_año_mes.csv
    all_rows = []

    for month in range(1, 13):
        # URL del formato correcto del Ayuntamiento de Madrid
        url = f"https://datos.madrid.es/egob/catalogo/212531-7749-calidad-aire-horario.csv"
        break

    # Usar la URL de descarga del portal de datos abiertos
    # Formato CSV horario del Ayuntamiento de Madrid
    download_url = f"https://datos.madrid.es/egob/catalogo/212531-7749-calidad-aire-horario.zip"

    try:
        r = requests.get(download_url, timeout=60)
        if r.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                for name in z.namelist():
                    if name.endswith('.csv'):
                        with z.open(name) as f:
                            df = pd.read_csv(f, sep=';', encoding='latin1')
                            all_rows.append(df)
            if all_rows:
                return pd.concat(all_rows, ignore_index=True)
    except Exception as e:
        print(f"    Error descargando {download_url}: {e}")

    return None


def parse_madrid_csv(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte el formato CSV del Ayuntamiento (columnas H01..H24)
    a formato largo (timestamp, estacion, magnitud, valor).
    """
    rows = []
    hour_cols = [c for c in df_raw.columns if c.upper().startswith('H') and c[1:].isdigit()]

    for _, row in df_raw.iterrows():
        try:
            station = int(row.get('ESTACION', row.get('estacion', 0)))
            magnitud = int(row.get('MAGNITUD', row.get('magnitud', 0)))
            year = int(row.get('ANO', row.get('ano', row.get('YEAR', 2023))))
            month = int(row.get('MES', row.get('mes', row.get('MONTH', 1))))
            day = int(row.get('DIA', row.get('dia', row.get('DAY', 1))))
        except (ValueError, TypeError):
            continue

        for hcol in hour_cols:
            try:
                hour = int(hcol[1:]) - 1  # H01 → hora 0
                val_col = hcol
                valid_col = 'V' + hcol[1:]

                val = row.get(val_col, np.nan)
                valid = row.get(valid_col, 'V')

                if str(valid).strip().upper() != 'V':
                    continue

                val = float(str(val).replace(',', '.'))
                if val < 0:
                    continue

                ts = datetime.datetime(year, month, day, hour)
                rows.append({
                    'timestamp': ts,
                    'station': station,
                    'magnitud': magnitud,
                    'valor': val,
                })
            except Exception:
                continue

    return pd.DataFrame(rows)


def fetch_openmeteo_weather(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    """Descarga datos meteorológicos horarios de Open-Meteo (sin API key)."""
    url = (
        f"https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={start}&end_date={end}"
        f"&hourly=temperature_2m,wind_speed_10m,wind_direction_10m"
        f"&wind_speed_unit=ms&timezone=Europe/Madrid"
    )
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        data = r.json()
        hourly = data.get('hourly', {})
        df = pd.DataFrame({
            'timestamp': pd.to_datetime(hourly['time']),
            'temp': hourly['temperature_2m'],
            'wind_speed': hourly['wind_speed_10m'],
            'wind_dir': hourly['wind_direction_10m'],
        })
        return df
    except Exception as e:
        print(f"  Open-Meteo error: {e}")
        return pd.DataFrame()


def generate_synthetic_data(n_stations: int, n_hours: int = 8760) -> pd.DataFrame:
    """
    Genera datos sintéticos realistas de calidad del aire para Madrid.
    Se usa si la descarga falla. Modela:
    - Patrones diarios (hora punta mañana/tarde)
    - Patrones semanales (más tráfico entre semana)
    - Estacionalidad anual
    - Correlación espacial entre estaciones cercanas
    - Variabilidad aleatoria
    """
    print("  Generando datos sintéticos realistas...")
    rng = np.random.default_rng(42)

    hours = pd.date_range(start="2023-01-01", periods=n_hours, freq="h")
    records = []

    # Perfil diario de NO2 (tráfico)
    hour_of_day = np.array([h.hour for h in hours])
    day_of_week = np.array([h.dayofweek for h in hours])
    day_of_year = np.array([h.dayofyear for h in hours])

    # Patrón horario: punta mañana (8h) y tarde (19h)
    no2_hourly = (
        15 * np.exp(-0.5 * ((hour_of_day - 8) / 2) ** 2) +   # punta mañana
        12 * np.exp(-0.5 * ((hour_of_day - 19) / 2) ** 2) +  # punta tarde
        5                                                       # base nocturna
    )
    # Reducción fin de semana
    weekend_mask = day_of_week >= 5
    no2_hourly[weekend_mask] *= 0.6

    # Estacionalidad anual (invierno peor)
    seasonal = 1.0 + 0.3 * np.cos(2 * np.pi * (day_of_year - 15) / 365)
    no2_base = no2_hourly * seasonal

    # Patrón PM10
    pm10_base = 15 + 0.4 * no2_base + 5 * np.sin(2 * np.pi * day_of_year / 365)

    # Patrón O3 (opuesto al NO2, más en verano y mediodía)
    o3_hourly_pattern = 20 * np.exp(-0.5 * ((hour_of_day - 13) / 3) ** 2)
    o3_seasonal = 1.0 + 0.5 * np.cos(2 * np.pi * (day_of_year - 180) / 365)
    o3_base = (25 + o3_hourly_pattern) * o3_seasonal - 0.3 * no2_base

    # Meteorología sintética
    temp_base = 13 + 9 * np.cos(2 * np.pi * (day_of_year - 200) / 365)
    temp_daily = 4 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)
    wind_speed = np.abs(rng.normal(3, 2, n_hours))
    wind_dir = rng.uniform(0, 360, n_hours)

    for i, station in enumerate(STATIONS):
        # Variación por estación (urbana vs periférica)
        is_peripheral = station['lat'] > 40.47 or station['lat'] < 40.37
        urban_factor = 0.7 if is_peripheral else 1.0

        noise_scale = 0.15
        no2 = np.maximum(0, no2_base * urban_factor + rng.normal(0, no2_base * noise_scale))
        pm10 = np.maximum(0, pm10_base * urban_factor + rng.normal(0, pm10_base * noise_scale))
        o3 = np.maximum(0, o3_base + rng.normal(0, 8))
        temp = temp_base + temp_daily + rng.normal(0, 1, n_hours)

        for j, ts in enumerate(hours):
            records.append({
                'timestamp': ts,
                'station': station['codigo'],
                'station_id': station['id'],
                'nombre': station['nombre'],
                'lat': station['lat'],
                'lon': station['lon'],
                'NO2': round(float(no2[j]), 1),
                'PM10': round(float(pm10[j]), 1),
                'O3': max(0, round(float(o3[j]), 1)),
                'temp': round(float(temp[j]), 1),
                'wind_speed': round(float(wind_speed[j]), 1),
                'wind_dir': round(float(wind_dir[j]), 1),
            })

    return pd.DataFrame(records)


# ── ICA ───────────────────────────────────────────────────────────────────────

def no2_to_ica(no2: float) -> float:
    """Convierte NO2 (µg/m³) a ICA escala española (0-500)."""
    breakpoints = [
        (0, 40, 0, 25),
        (40, 90, 25, 50),
        (90, 120, 50, 75),
        (120, 230, 75, 100),
        (230, 340, 100, 150),
        (340, float('inf'), 150, 500),
    ]
    for lo_c, hi_c, lo_i, hi_i in breakpoints:
        if lo_c <= no2 < hi_c:
            if hi_c == float('inf'):
                return min(500, lo_i + (no2 - lo_c))
            return lo_i + (no2 - lo_c) / (hi_c - lo_c) * (hi_i - lo_i)
    return 0.0


def pm10_to_ica(pm10: float) -> float:
    """Convierte PM10 (µg/m³) a ICA."""
    breakpoints = [
        (0, 20, 0, 25),
        (20, 35, 25, 50),
        (35, 50, 50, 75),
        (50, 100, 75, 100),
        (100, 150, 100, 150),
        (150, float('inf'), 150, 500),
    ]
    for lo_c, hi_c, lo_i, hi_i in breakpoints:
        if lo_c <= pm10 < hi_c:
            if hi_c == float('inf'):
                return min(500, lo_i + (pm10 - lo_c))
            return lo_i + (pm10 - lo_c) / (hi_c - lo_c) * (hi_i - lo_i)
    return 0.0


def o3_to_ica(o3: float) -> float:
    """Convierte O3 (µg/m³) a ICA."""
    breakpoints = [
        (0, 60, 0, 25),
        (60, 120, 25, 50),
        (120, 180, 50, 75),
        (180, 240, 75, 100),
        (240, 380, 100, 150),
        (380, float('inf'), 150, 500),
    ]
    for lo_c, hi_c, lo_i, hi_i in breakpoints:
        if lo_c <= o3 < hi_c:
            if hi_c == float('inf'):
                return min(500, lo_i + (o3 - lo_c))
            return lo_i + (o3 - lo_c) / (hi_c - lo_c) * (hi_i - lo_i)
    return 0.0


def compute_ica(no2: float, pm10: float, o3: float) -> float:
    """ICA global = máximo de los contaminantes individuales."""
    return max(no2_to_ica(no2), pm10_to_ica(pm10), o3_to_ica(o3))


# ── Grafo de estaciones ───────────────────────────────────────────────────────

def build_adjacency_matrix(stations: list) -> np.ndarray:
    """
    Construye la matriz de adyacencia con kernel gaussiano sobre distancia.
    A[i,j] = exp(-dist²/σ²) si dist < umbral, else 0.
    Normalizada por fila: L = D^{-1} A.
    """
    n = len(stations)
    coords = np.array([[s['lat'], s['lon']] for s in stations])
    A = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dist = math.sqrt(
                (coords[i, 0] - coords[j, 0]) ** 2 +
                (coords[i, 1] - coords[j, 1]) ** 2
            )
            if dist < DIST_THRESHOLD:
                A[i, j] = math.exp(-dist ** 2 / DIST_SIGMA ** 2)

    # Normalización por fila (random walk)
    row_sums = A.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    L = A / row_sums
    return L


# ── Modelo DCRNN-lite ─────────────────────────────────────────────────────────

class DCGRUCell(nn.Module):
    """
    Celda GRU con Diffusion Graph Convolution.
    Implementación pura PyTorch (sin PyG).
    """

    def __init__(self, input_dim: int, hidden_dim: int, k_hops: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.k_hops = k_hops

        # Cada hop añade una dimensión extra
        conv_input_dim = input_dim * (k_hops + 1)
        conv_hidden_dim = hidden_dim * (k_hops + 1)

        # Gates r (reset) y u (update)
        self.linear_r = nn.Linear(conv_input_dim + conv_hidden_dim, hidden_dim)
        self.linear_u = nn.Linear(conv_input_dim + conv_hidden_dim, hidden_dim)
        # Candidato
        self.linear_c = nn.Linear(conv_input_dim + conv_hidden_dim, hidden_dim)

    def _diffuse(self, x: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        """
        Aplica K potencias de L al input x y concatena.
        x: [batch, N, F]
        L: [N, N]
        returns: [batch, N, F*(K+1)]
        """
        out = [x]
        Lk = L
        xk = x
        for _ in range(self.k_hops):
            # bmm: L es compartida entre batches → einsum
            xk = torch.einsum('nm,bmf->bnf', Lk, x)
            out.append(xk)
            Lk = Lk @ L
        return torch.cat(out, dim=-1)  # [batch, N, F*(K+1)]

    def forward(
        self,
        x: torch.Tensor,     # [batch, N, input_dim]
        h: torch.Tensor,     # [batch, N, hidden_dim]
        L: torch.Tensor,     # [N, N]
    ) -> torch.Tensor:
        x_diff = self._diffuse(x, L)   # [batch, N, input_dim*(K+1)]
        h_diff = self._diffuse(h, L)   # [batch, N, hidden_dim*(K+1)]

        xh = torch.cat([x_diff, h_diff], dim=-1)

        r = torch.sigmoid(self.linear_r(xh))
        u = torch.sigmoid(self.linear_u(xh))

        # Aplicar reset gate a h (no a h_diff) y luego difundir
        rh_diff = self._diffuse(r * h, L)  # [batch, N, hidden_dim*(K+1)]
        xh_c = torch.cat([x_diff, rh_diff], dim=-1)
        c = torch.tanh(self.linear_c(xh_c))

        h_new = (1 - u) * h + u * c
        return h_new


class DCRNN(nn.Module):
    """
    Diffusion Convolutional Recurrent Neural Network (lite).
    Encoder: T pasos de DCGRU.
    Decoder: lineal sobre el estado final.
    """

    def __init__(self, n_stations: int, input_dim: int, hidden_dim: int,
                 n_layers: int, k_hops: int, horizon: int):
        super().__init__()
        self.n_stations = n_stations
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.horizon = horizon

        self.cells = nn.ModuleList([
            DCGRUCell(
                input_dim if i == 0 else hidden_dim,
                hidden_dim,
                k_hops
            )
            for i in range(n_layers)
        ])

        # Decoder: estado oculto → predicción de horizon pasos
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, horizon),
        )

    def forward(
        self,
        x: torch.Tensor,   # [batch, T, N, F]
        L: torch.Tensor,   # [N, N]
    ) -> torch.Tensor:
        batch, T, N, F = x.shape
        device = x.device

        # Estado inicial
        h = [torch.zeros(batch, N, self.hidden_dim, device=device)
             for _ in range(self.n_layers)]

        # Encoder
        for t in range(T):
            inp = x[:, t, :, :]  # [batch, N, F]
            for i, cell in enumerate(self.cells):
                h[i] = cell(inp, h[i], L)
                inp = h[i]

        # Decoder sobre el último estado oculto
        # h[-1]: [batch, N, hidden_dim]
        out = self.decoder(h[-1])  # [batch, N, horizon]
        return out


# ── Dataset ───────────────────────────────────────────────────────────────────

def build_feature_matrix(df: pd.DataFrame, station_map: dict) -> tuple:
    """
    Construye el array de features con forma [T_total, N_stations, n_features].
    Features por estación: [NO2, PM10, O3, wind_speed, wind_sin, wind_cos, temp, hour_sin, hour_cos]
    """
    # Timestamps únicos ordenados
    timestamps = sorted(df['timestamp'].unique())
    T = len(timestamps)
    N = len(STATIONS)
    ts_idx = {ts: i for i, ts in enumerate(timestamps)}

    n_features = len(POLLUTANTS) + len(META_FEATURES)
    X = np.zeros((T, N, n_features), dtype=np.float32)
    # Máscara de validez (1 = dato real, 0 = imputado)
    valid = np.zeros((T, N), dtype=np.float32)

    for _, row in df.iterrows():
        t = ts_idx.get(row['timestamp'])
        s = station_map.get(row.get('station', row.get('codigo', -1)))
        if t is None or s is None:
            continue

        # Contaminantes
        for k, pol in enumerate(POLLUTANTS):
            val = row.get(pol, np.nan)
            if not np.isnan(val):
                X[t, s, k] = val

        # Meteorología
        hour = row['timestamp'].hour
        X[t, s, len(POLLUTANTS)]     = row.get('wind_speed', 0)
        X[t, s, len(POLLUTANTS) + 1] = math.sin(math.radians(row.get('wind_dir', 0)))
        X[t, s, len(POLLUTANTS) + 2] = math.cos(math.radians(row.get('wind_dir', 0)))
        X[t, s, len(POLLUTANTS) + 3] = row.get('temp', 15)
        X[t, s, len(POLLUTANTS) + 4] = math.sin(2 * math.pi * hour / 24)
        X[t, s, len(POLLUTANTS) + 5] = math.cos(2 * math.pi * hour / 24)
        valid[t, s] = 1

    # ICA target
    ica_target = np.zeros((T, N), dtype=np.float32)
    for t in range(T):
        for s in range(N):
            ica_target[t, s] = compute_ica(X[t, s, 0], X[t, s, 1], X[t, s, 2])

    return X, ica_target, timestamps


def create_sequences(X: np.ndarray, y: np.ndarray,
                     lookback: int, horizon: int) -> tuple:
    """Crea secuencias de entrenamiento (X_seq, y_seq)."""
    T = len(X)
    xs, ys = [], []
    for i in range(T - lookback - horizon):
        xs.append(X[i:i + lookback])
        ys.append(y[i + lookback:i + lookback + horizon])
    return np.array(xs), np.array(ys)


# ── Entrenamiento ─────────────────────────────────────────────────────────────

def train():
    print("=" * 60)
    print("DCRNN-lite — Predicción de Calidad del Aire (Madrid)")
    print("=" * 60)

    # 1. Datos
    print("\n[1/5] Cargando datos...")
    df = generate_synthetic_data(n_stations=len(STATIONS), n_hours=2160)  # 3 meses
    print(f"  {len(df):,} registros · {df['timestamp'].nunique():,} timestamps · {len(STATIONS)} estaciones")

    station_map = {s['codigo']: i for i, s in enumerate(STATIONS)}

    # 2. Features
    print("\n[2/5] Construyendo features...")
    X_raw, y_ica, timestamps = build_feature_matrix(df, station_map)

    # Normalización
    T, N, F = X_raw.shape
    X_flat = X_raw.reshape(-1, F)
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X_flat).reshape(T, N, F).astype(np.float32)

    # También normalizar y (ICA: 0-500 → z-score)
    y_mean = float(y_ica.mean())
    y_std  = float(y_ica.std()) or 1.0

    y_norm = ((y_ica - y_mean) / y_std).astype(np.float32)

    # Secuencias
    X_seq, y_seq = create_sequences(X_norm, y_norm, LOOKBACK, HORIZON)
    print(f"  Secuencias: {len(X_seq):,} · X {X_seq.shape} · y {y_seq.shape}")

    # Split train/val
    n_train = int(0.85 * len(X_seq))
    X_tr, X_val = X_seq[:n_train], X_seq[n_train:]
    y_tr, y_val = y_seq[:n_train], y_seq[n_train:]

    # 3. Grafo
    print("\n[3/5] Construyendo grafo de estaciones...")
    L = build_adjacency_matrix(STATIONS)
    print(f"  Matriz de adyacencia: {L.shape} · densidad: {(L > 0).mean():.1%}")

    # 4. Modelo
    print("\n[4/5] Entrenando DCRNN-lite...")
    device = torch.device('cpu')
    L_tensor = torch.tensor(L, dtype=torch.float32, device=device)

    model = DCRNN(
        n_stations=N,
        input_dim=F,
        hidden_dim=HIDDEN_DIM,
        n_layers=N_LAYERS,
        k_hops=K_HOPS,
        horizon=HORIZON,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parámetros: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.HuberLoss()

    X_tr_t  = torch.tensor(X_tr,  dtype=torch.float32, device=device)
    y_tr_t  = torch.tensor(y_tr,  dtype=torch.float32, device=device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device)

    best_val_loss = float('inf')
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(X_tr_t))
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, len(X_tr_t), BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            xb = X_tr_t[idx]   # [B, T, N, F]
            yb = y_tr_t[idx]   # [B, N, H]

            optimizer.zero_grad()
            pred = model(xb, L_tensor)   # [B, N, H]
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        if epoch % 10 == 0 or epoch == EPOCHS:
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_t, L_tensor)
                val_loss = criterion(val_pred, y_val_t).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

            print(f"  Epoch {epoch:3d}/{EPOCHS} | train_loss={epoch_loss/n_batches:.4f} | val_loss={val_loss:.4f}")

    # Cargar mejor estado
    if best_state:
        model.load_state_dict(best_state)

    # 5. Evaluación
    print("\n[5/5] Evaluando y guardando artifacts...")
    model.eval()
    with torch.no_grad():
        val_pred_t = model(X_val_t, L_tensor)

    val_pred_np = val_pred_t.numpy() * y_std + y_mean
    val_true_np = y_val_t.numpy() * y_std + y_mean

    # Métricas en escala ICA (promedio sobre estaciones y horizonte)
    mae  = float(mean_absolute_error(val_true_np.flatten(), val_pred_np.flatten()))
    r2   = float(r2_score(val_true_np.flatten(), val_pred_np.flatten()))
    rmse = float(np.sqrt(np.mean((val_true_np - val_pred_np) ** 2)))

    print(f"  MAE ICA:  {mae:.2f}")
    print(f"  RMSE ICA: {rmse:.2f}")
    print(f"  R²:       {r2:.4f}")

    # Guardar model
    torch.save({
        'state_dict': model.state_dict(),
        'n_stations': N,
        'input_dim':  F,
        'hidden_dim': HIDDEN_DIM,
        'n_layers':   N_LAYERS,
        'k_hops':     K_HOPS,
        'horizon':    HORIZON,
        'lookback':   LOOKBACK,
        'y_mean':     y_mean,
        'y_std':      y_std,
    }, ARTIFACTS / 'dcrnn_model.pt')

    # Guardar scaler
    joblib.dump(scaler, ARTIFACTS / 'scaler.joblib')

    # Guardar adjacency
    np.save(ARTIFACTS / 'adj_matrix.npy', L)

    # Guardar metadata de estaciones
    with open(ARTIFACTS / 'stations.json', 'w', encoding='utf-8') as f:
        json.dump(STATIONS, f, ensure_ascii=False, indent=2)

    # Guardar metadata del modelo
    metadata = {
        'mae_ica':     round(mae, 2),
        'rmse_ica':    round(rmse, 2),
        'r2':          round(r2, 4),
        'n_params':    n_params,
        'n_stations':  N,
        'input_dim':   F,
        'lookback':    LOOKBACK,
        'horizon':     HORIZON,
        'epochs':      EPOCHS,
        'hidden_dim':  HIDDEN_DIM,
        'n_layers':    N_LAYERS,
        'k_hops':      K_HOPS,
        'pollutants':  POLLUTANTS,
        'features':    POLLUTANTS + META_FEATURES,
        'dataset':     'Sintético (basado en Red Vigilancia Calidad Aire Madrid)',
        'ciudad':      'Madrid',
        'n_sequences': len(X_seq),
        'fecha_entrenamiento': datetime.datetime.now().isoformat(),
    }
    with open(ARTIFACTS / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\nArtifacts guardados en artifacts/:")
    for p in sorted(ARTIFACTS.iterdir()):
        size_kb = p.stat().st_size / 1024
        print(f"  {p.name} ({size_kb:.1f} KB)")

    print(f"\nEntrenamiento completado.")
    print(f"  MAE ICA = {mae:.1f} | R² = {r2:.4f}")
    return metadata


if __name__ == '__main__':
    train()
