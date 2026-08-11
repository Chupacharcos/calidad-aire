# Predicción de Calidad del Aire — DCRNN-lite

Sistema de predicción del **Índice de Calidad del Aire (ICA) a 24 horas** mediante una Red Neuronal Recurrente con Convolución Difusiva (DCRNN-lite). Modela 24 estaciones de monitorización de Madrid como un grafo espacial y propaga señales de contaminación entre estaciones vecinas.

Demo en producción: [adrianmoreno-dev.com/demo/prediccion-calidad-aire](https://adrianmoreno-dev.com/demo/prediccion-calidad-aire)

---

<!-- LOOP-MAP:START (generado por `php artisan project:loop readme` — no editar a mano) -->

## El bucle que cierra

<p align="center"><img src="https://adrianmoreno-dev.com/bucle/prediccion-calidad-aire.svg" alt="Mapa del bucle de Predicción de Calidad del Aire" width="900"></p>

**Para** quien vigila la calidad del aire de Madrid · **Cada día**

| Etapa | Qué pasa | Quién |
|---|---|---|
| **1. Disparador** | Necesito saber si mañana habrá un episodio de contaminación en la ciudad. | persona |
| **2. Acción** | Propaga las lecturas entre las 24 estaciones vecinas como un grafo y predice el índice de cada una a 24 horas. | software |
| **3. Medición** | La categoría ICA prevista por estación, con un error medio de ±2,8 puntos de índice. | software |
| **4. Decisión** | Decido si aviso, si se activa el protocolo o si no hace falta hacer nada. | persona |

### Lo que no hace

- Solo cubre Madrid: el grafo son las 24 estaciones de la ciudad, no otras redes.
- No pasa de 24 horas: el horizonte de la predicción es un día.
- No mide nada: consume las lecturas de las estaciones, no sustituye a un sensor.

### Por qué está construido así

- **Las estaciones, como grafo** en vez de una serie temporal independiente por estación — La contaminación viaja entre estaciones vecinas. Una serie aislada no ve venir el episodio que ya está en la estación de al lado.
- **Servir en NGSI-LD y GeoJSON** en vez de un JSON con formato propio — Son los formatos que ya hablan las plataformas de ciudad y los visores de mapas, así que se conecta sin escribir un adaptador.

<!-- LOOP-MAP:END -->

## Resultados

| Métrica | Valor |
|---------|-------|
| **F1-score (macro)** | **0.87** |
| **MCC** | **0.74** |
| **Exactitud categoría ICA** | **97%** |
| **R² por estación** | 0.84 |
| **MAE** | ±2.8 ICA |
| Precision categoría "Moderada" | 0.80 |
| Recall categoría "Moderada" | 0.72 |
| Muestras de evaluación | 21.024 (24 est. × tiempo) |
| Clases ICA | Buena (93.5%) / Moderada (6.5%) |

> **MCC ≥ 0.7 = excelente** incluso con clases muy desbalanceadas. El modelo clasifica correctamente el 97% de las horas y detecta el 72% de los episodios de contaminación moderada.

---

## Arquitectura

```
24 estaciones Madrid (grafo espacial)
         │
         ▼
┌──────────────────────────────────────┐
│  Features de entrada (10 por nodo)   │
│  · NO₂, PM10, O₃                    │
│  · Velocidad de viento               │
│  · sin/cos dirección del viento      │
│  · Temperatura                       │
│  · sin/cos hora del día              │
│  · ICA actual                        │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│  Grafo de adyacencia                 │
│  Kernel gaussiano sobre distancia    │
│  euclidiana, normalizado por fila    │
│  (random walk diffusion)             │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│  DCGRUCell (Diffusion Graph Conv.)   │
│  K=1 hop de difusión por capa        │
│  2 capas, hidden_dim=32              │
└──────────────┬───────────────────────┘
               │
               ▼
    Predicción H=1 (siguiente hora)
    + autoregresión iterativa → 24h
               │
               ▼
    ICA predicho por estación (0-500)
    → Categoría ICA (Buena / Moderada / Dañina)
```

### ¿Por qué DCRNN y no LSTM independiente por estación?

El DCRNN captura la **dinámica de dispersión atmosférica**: si la estación A detecta un pico de NO₂, la estación B (a 2 km en dirección del viento) lo verá 30-60 minutos después. Un LSTM por estación no puede modelar esto. La coherencia espacial (estaciones vecinas sin predicciones contradictorias) es una propiedad garantizada por el grafo.

---

## Categorías ICA (Índice de Calidad del Aire)

| Score ICA | Categoría | Color |
|-----------|-----------|-------|
| 0 – 50 | Buena | Verde |
| 51 – 100 | Moderada | Amarillo |
| 101 – 150 | Dañina para grupos sensibles | Naranja |
| 151 – 200 | Dañina | Rojo |
| > 200 | Muy dañina / Peligrosa | Marrón/Morado |

---

## Estructura del proyecto

```
calidad-aire/
├── train.py          # Entrenamiento DCRNN-lite (ejecutar offline)
├── router.py         # Endpoints FastAPI (/ml/calidad-aire/*)
├── api.py            # App FastAPI standalone (puerto 8091)
└── artifacts/        # Modelo entrenado (excluido de git)
    ├── dcrnn_model.pt
    ├── scaler.joblib
    ├── graph_adj.npy
    └── metadata.json
```

---

## Endpoints REST

| Método | Ruta | Descripción |
|--------|------|-------------|
| `GET` | `/ml/calidad-aire/prediccion` | ICA predicho 24h para todas las estaciones |
| `GET` | `/ml/calidad-aire/estaciones` | Lista de estaciones con coordenadas |
| `GET` | `/ml/calidad-aire/stats` | Métricas del modelo |

### Respuesta `/prediccion` (extracto)

```json
{
  "ok": true,
  "estaciones": [
    {
      "id": "est_001",
      "nombre": "Plaza España",
      "lat": 40.4231, "lng": -3.7120,
      "ica_actual": 42,
      "categoria_actual": "Buena",
      "forecast_24h": [
        {"hora": 1, "ica": 44, "categoria": "Buena"},
        {"hora": 6, "ica": 67, "categoria": "Moderada"},
        {"hora": 12, "ica": 38, "categoria": "Buena"}
      ]
    }
  ],
  "modelo": "DCRNN-lite K=1, F1=0.87, MCC=0.74"
}
```

---

## Entrenamiento

```bash
cd /var/www/calidad-aire
source /var/www/chatbot/venv/bin/activate
python3 train.py
```

Genera el grafo de adyacencia geográfica, entrena el DCRNN-lite con datos del servicio CAMS Reanalysis de Open-Meteo (2022-2023, 24 estaciones Madrid) y guarda los artifacts. Requiere PyTorch.

## Arranque del servicio

```bash
uvicorn api:app --host 127.0.0.1 --port 8091 --reload
```

---

## Datos

- **Fuente:** Open-Meteo CAMS Reanalysis (Copernicus Atmosphere Monitoring Service)
- **Periodo:** 2022-2023 (8.760 horas)
- **Variables:** NO₂, PM10, O₃, velocidad y dirección de viento, temperatura
- **Estaciones:** 24 puntos de Madrid (coordenadas reales)

---

## Stack técnico

- **Python 3.12** · **PyTorch 2.10** (DCGRUCell custom puro PyTorch)
- **NumPy / scikit-learn** — preprocesamiento y métricas
- **FastAPI / Uvicorn** — API REST
- **joblib** — serialización del scaler
- **Leaflet.js** — mapa interactivo (frontend, en repositorio del portfolio)
- **Open-Meteo CAMS** — datos atmosféricos

---

*Parte del portafolio de proyectos IA/ML — [adrianmoreno-dev.com](https://adrianmoreno-dev.com)*


## Integración, datos y licencia

**Licencia:** MIT (ver [LICENSE](LICENSE)) — uso libre, incluido comercial,
manteniendo el aviso de copyright. Sin garantía ni soporte incluidos.

### Formatos estándar de salida

Además del JSON propio, la predicción se sirve en dos formatos que las
plataformas del sector ya consumen, para no obligar a nadie a programar contra
un esquema propietario:

| Formato | Endpoint | Para qué |
|---|---|---|
| **NGSI-LD** (ETSI CIM 009, modelo `AirQualityObserved` de FIWARE Smart Data Models) | `GET /ml/calidad-aire/ngsi-ld` | Volcado directo a un Context Broker (Orion-LD, Scorpio, Stellio) de una plataforma de ciudad inteligente |
| **GeoJSON** (RFC 7946) | `GET /ml/calidad-aire/geojson` | Cargar en QGIS, ArcGIS, Leaflet o Mapbox sin conversión |

```bash
# Alta/actualización en un Context Broker NGSI-LD
curl -s http://localhost:8091/ml/calidad-aire/ngsi-ld?ciudad=madrid > entidades.json
curl -X POST "$BROKER/ngsi-ld/v1/entityOperations/upsert" \
     -H 'Content-Type: application/ld+json' -d @entidades.json

# Capa GeoJSON directa en un mapa
curl "http://localhost:8091/ml/calidad-aire/geojson?ciudad=madrid" -o estaciones.geojson
```

Ambos se generan a partir de la misma predicción que sirve `/estaciones`, así que
no puede haber discrepancias entre formatos. La predicción a 24 h viaja en el
atributo propio `airQualityIndexForecast` (el modelo estándar no la contempla),
declarado como tal para que quede claro qué es estándar y qué es extensión.

### Home Assistant (MQTT)

`mqtt_publish.py` publica la predicción en un broker MQTT con
**autodescubrimiento de Home Assistant**: cada estación aparece como sensor sin
tocar `configuration.yaml`.

```bash
pip install paho-mqtt          # dependencia sólo de este script

python mqtt_publish.py --host 192.168.1.50 --ciudad madrid
python mqtt_publish.py --host broker --port 8883 --tls --user ha --password ***
python mqtt_publish.py --host x --dry-run     # ver qué se enviaría, sin publicar
```

| Topic | Contenido |
|---|---|
| `homeassistant/sensor/calidad_aire_<ciudad>_<estación>/config` | Config de descubrimiento (*retained*) |
| `calidad_aire/<ciudad>/<estación>/state` | `ica_actual`, `categoria`, `ica_max_24h`, coordenadas |

Las estaciones de una ciudad se agrupan bajo un mismo *device* en la UI de Home
Assistant. Con eso se pueden montar automatismos del tipo «cierra la ventilación
si `ica_max_24h` supera X».

**Es un script, no un hilo del servicio, a propósito:** esta API arranca bajo
demanda y se apaga tras 30 min sin tráfico, así que un publicador continuo
integrado moriría con ella. Como script, se programa por cron en la
infraestructura del integrador, junto a su broker.

### Tratamiento de datos

No hay datos personales en ningún punto: el servicio trabaja con estaciones de
medición y valores agregados. No se almacenan peticiones ni se registra quién
consulta.

**Qué sale del servidor:** nada. **Este proyecto no usa ningún proveedor de IA
externo** — el modelo (DCRNN-lite en PyTorch) se ejecuta localmente. Funciona
sin conexión a internet y sin ninguna clave de API.

### Despliegue propio y costes

El repositorio es la aplicación completa (FastAPI + systemd + modelo entrenado).
Se despliega en infraestructura propia sin dependencias SaaS. El código es
gratuito (MIT); los costes son la infraestructura y el mantenimiento, a cargo de
quien lo despliega — el autor no ofrece soporte ni consultoría.
