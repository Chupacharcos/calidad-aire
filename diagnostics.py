from pathlib import Path
from datetime import datetime
from fastapi import APIRouter

router = APIRouter(prefix="/ml", tags=["diagnostics"])

ARTIFACTS = Path(__file__).parent / "artifacts"

@router.get("/diagnostics")
def diagnostics():
    """Devuelve metadata del modelo y estado del servicio."""
    info = {
        "service": "ml-calidad-aire",
        "version": "1.0",
        "model_arch": "DCRNN-lite (K=2 diffusion hops, 2 layers, hidden=32)",
        "horizon_hours": 24,
        "ciudades": ["madrid", "barcelona", "valencia"],
        "features_input": ["NO2", "PM10", "O3", "wind", "temp", "hour_sin", "hour_cos"],
        "artifacts_present": ARTIFACTS.is_dir(),
    }
    if ARTIFACTS.is_dir():
        files = sorted([f.name for f in ARTIFACTS.iterdir() if f.is_file()])
        info["artifact_files"] = files[:10]
        if files:
            newest = max(ARTIFACTS.iterdir(), key=lambda p: p.stat().st_mtime)
            info["last_training_at"] = datetime.fromtimestamp(
                newest.stat().st_mtime
            ).isoformat()
    return info