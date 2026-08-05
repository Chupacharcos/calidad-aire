"""
Exportadores en formatos estándar para integradores.

Responde a la pregunta "¿cómo lo integro con lo que ya tengo?" sin obligar a
nadie a programar contra un JSON propietario:

  - NGSI-LD (ETSI CIM 009) con el modelo AirQualityObserved de FIWARE Smart
    Data Models: es lo que consumen los Context Brokers (Orion-LD, Scorpio,
    Stellio) de las plataformas de ciudad inteligente.
  - GeoJSON (RFC 7946): lo lee cualquier GIS o mapa web (QGIS, ArcGIS,
    Leaflet, Mapbox, OpenLayers) sin escribir una línea de código.

Ambos se generan a partir de la MISMA predicción que sirve /estaciones, así que
no hay riesgo de que un formato diga una cosa y otro diga otra.
"""
from __future__ import annotations

import datetime

# Contexto JSON-LD oficial de FIWARE Smart Data Models
NGSI_LD_CONTEXT = [
    "https://raw.githubusercontent.com/smart-data-models/dataModel.Environment/master/context.jsonld",
    "https://uri.etsi.org/ngsi-ld/v1/ngsi-ld-core-context.jsonld",
]


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def to_ngsi_ld(estaciones: list[dict], ciudad: str) -> list[dict]:
    """Lista de entidades NGSI-LD `AirQualityObserved`, una por estación.

    Se puede volcar tal cual a un Context Broker:
        curl -X POST <broker>/ngsi-ld/v1/entityOperations/upsert \\
             -H 'Content-Type: application/ld+json' -d @entidades.json
    """
    observed_at = _now_iso()
    entities = []

    for st in estaciones:
        # El id de una entidad NGSI-LD debe ser una URN según el spec.
        urn = f"urn:ngsi-ld:AirQualityObserved:{ciudad}:{st.get('id', st.get('codigo', 'NA'))}"
        entity = {
            "id": urn,
            "type": "AirQualityObserved",
            "@context": NGSI_LD_CONTEXT,
            "dateObserved": {"type": "Property", "value": {"@type": "DateTime", "@value": observed_at}},
            "location": {
                "type": "GeoProperty",
                "value": {"type": "Point", "coordinates": [st["lon"], st["lat"]]},  # GeoJSON: lon, lat
            },
            "name": {"type": "Property", "value": st.get("nombre", "")},
            "areaServed": {"type": "Property", "value": ciudad},
        }

        # Índice de calidad del aire actual (atributo estándar del modelo).
        if st.get("ica_actual") is not None:
            entity["airQualityIndex"] = {"type": "Property", "value": st["ica_actual"]}
            entity["airQualityLevel"] = {"type": "Property", "value": st.get("ica_categoria", "")}

        # La predicción NO es parte del modelo estándar: va como atributo propio
        # y con `observedAt` para que quede claro que es un valor derivado.
        pred = st.get("ica_prediccion_24h")
        if pred:
            entity["airQualityIndexForecast"] = {
                "type": "Property",
                "value": list(pred),
                "unitCode": "P1",  # UN/CEFACT: índice adimensional
                "observedAt": observed_at,
                "forecastHorizonHours": {"type": "Property", "value": len(pred)},
            }
        entities.append(entity)

    return entities


def to_geojson(estaciones: list[dict], ciudad: str) -> dict:
    """FeatureCollection RFC 7946. Las propiedades se mantienen planas a
    propósito: así QGIS y Leaflet pueden colorear por `ica_actual` sin
    transformaciones previas."""
    features = []
    for st in estaciones:
        props = {k: v for k, v in st.items() if k not in ("lat", "lon")}
        # Las listas largas rompen la tabla de atributos de algunos GIS.
        pred = props.pop("ica_prediccion_24h", None)
        if pred:
            props["ica_prediccion_max"] = max(pred)
            props["ica_prediccion_horas"] = len(pred)
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [st["lon"], st["lat"]]},
            "properties": props,
        })

    return {
        "type": "FeatureCollection",
        "features": features,
        # Metadatos fuera del spec pero admitidos como miembros extra.
        "metadata": {"ciudad": ciudad, "generado": _now_iso(), "fuente": "prediccion-calidad-aire"},
    }
