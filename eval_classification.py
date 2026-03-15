import torch, joblib, numpy as np, json, sys
from pathlib import Path
from sklearn.metrics import f1_score, matthews_corrcoef, classification_report

ARTIFACTS = Path("/var/www/calidad-aire/artifacts")
sys.path.insert(0, "/var/www/calidad-aire")

from train import (
    DCRNN, create_sequences, download_real_data, generate_synthetic_data,
    build_feature_matrix, STATIONS, compute_ica
)

# ── Load metadata & artifacts ──────────────────────────────────────────────────
meta   = json.loads((ARTIFACTS / "metadata.json").read_text())
scaler = joblib.load(ARTIFACTS / "scaler.joblib")

LOOKBACK   = meta["lookback"]
HORIZON    = meta["horizon"]
N_ST       = meta["n_stations"]
INPUT_DIM  = meta["input_dim"]
HIDDEN_DIM = meta["hidden_dim"]
N_LAYERS   = meta["n_layers"]
K_HOPS     = meta["k_hops"]

# ── Adjacency matrix ───────────────────────────────────────────────────────────
adj   = np.load(ARTIFACTS / "adj_matrix.npy")
D_inv = np.diag(1.0 / (adj.sum(axis=1) + 1e-8))
L     = D_inv @ adj
L_t   = torch.tensor(L, dtype=torch.float32)

# ── Load model ─────────────────────────────────────────────────────────────────
model = DCRNN(N_ST, INPUT_DIM, HIDDEN_DIM, N_LAYERS, K_HOPS, HORIZON)
ckpt  = torch.load(ARTIFACTS / "dcrnn_model.pt", map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["state_dict"])
model.eval()
y_mean = ckpt["y_mean"]
y_std  = ckpt["y_std"]

# ── Load data ──────────────────────────────────────────────────────────────────
print("Cargando datos...")
df = download_real_data(n_hours_target=8760)
if df is None or len(df) < 5000:
    print("  Fallback a datos sintéticos...")
    df = generate_synthetic_data(n_stations=len(STATIONS), n_hours=8760)

print(f"  {len(df):,} registros · {df['timestamp'].nunique():,} timestamps")

# ── Build feature matrix (same as train.py) ────────────────────────────────────
station_map = {s['codigo']: i for i, s in enumerate(STATIONS)}
X_raw, y_raw, timestamps = build_feature_matrix(df, station_map)
print(f"  X_raw shape: {X_raw.shape} | y_raw shape: {y_raw.shape}")

# ── Normalize ──────────────────────────────────────────────────────────────────
T, N, Fv = X_raw.shape
X_norm = scaler.transform(X_raw.reshape(-1, Fv)).reshape(T, N, Fv)

# ── Create sequences ───────────────────────────────────────────────────────────
X_seq, y_seq = create_sequences(X_norm, y_raw, LOOKBACK, HORIZON, stride=4)
print(f"  X_seq shape: {X_seq.shape} | y_seq shape: {y_seq.shape}")

n_train = int(0.8 * len(X_seq))
X_val = X_seq[n_train:]
y_val = y_seq[n_train:]
print(f"  Val sequences: {len(X_val):,}")

# ── Inference ──────────────────────────────────────────────────────────────────
all_pred, all_true = [], []
BATCH = 64
with torch.no_grad():
    for i in range(0, len(X_val), BATCH):
        xb = torch.tensor(X_val[i:i+BATCH], dtype=torch.float32)
        pb = model(xb, L_t).numpy()          # (B, N_st, horizon)
        pred_ica = pb * y_std + y_mean
        pred_ica = np.clip(pred_ica, 0, 500)
        all_pred.append(pred_ica)
        all_true.append(y_val[i:i+BATCH])

pred_all = np.concatenate(all_pred, axis=0)   # (N_val, N_st, H)
true_all = np.concatenate(all_true, axis=0)

# ── ICA category mapping ───────────────────────────────────────────────────────
def ica_category(ica):
    if ica <= 50:    return 0  # Buena
    elif ica <= 100: return 1  # Moderada
    elif ica <= 150: return 2  # Dañina sensible
    elif ica <= 200: return 3  # Dañina
    else:            return 4  # Muy dañina / Peligrosa

pred_flat = pred_all.flatten()
true_flat = true_all.flatten()

pred_cat = np.array([ica_category(v) for v in pred_flat])
true_cat = np.array([ica_category(v) for v in true_flat])

# ── Results ────────────────────────────────────────────────────────────────────
cats = ["Buena", "Moderada", "Dañina sensible", "Dañina", "Muy dañina"]

print("\n=== Calidad del Aire — Clasificación por Categoría ICA ===")
print(f"Total muestras: {len(pred_cat):,}")

print("\nDistribución real de categorías:")
for i, name in enumerate(cats):
    count = (true_cat == i).sum()
    print(f"  {name}: {count:,} ({count/len(true_cat)*100:.1f}%)")

f1_macro    = f1_score(true_cat, pred_cat, average='macro',    zero_division=0)
f1_weighted = f1_score(true_cat, pred_cat, average='weighted', zero_division=0)
mcc         = matthews_corrcoef(true_cat, pred_cat)

print(f"\nF1-score (macro)    : {f1_macro:.4f}")
print(f"F1-score (weighted) : {f1_weighted:.4f}")
print(f"MCC                 : {mcc:.4f}")
print("\nClassification report:")
present_labels = sorted(set(true_cat) | set(pred_cat))
present_names  = [cats[i] for i in present_labels]
print(classification_report(true_cat, pred_cat, labels=present_labels,
                            target_names=present_names, zero_division=0))
