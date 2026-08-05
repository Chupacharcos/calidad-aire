#!/usr/bin/env python3
"""
Publica la predicción de calidad del aire en un broker MQTT, con autodescubrimiento
de Home Assistant.

Por qué un script y no un hilo dentro de la API: este servicio arranca bajo
demanda y se apaga tras 30 min sin tráfico, así que un publicador continuo
integrado moriría con él. Como script, lo lanza el integrador por cron en SU
infraestructura, apuntando a SU broker — que es donde tiene sentido que viva.

Publica dos cosas por estación:

  - **Config de descubrimiento** (retained, en `homeassistant/sensor/.../config`):
    Home Assistant crea la entidad solo, sin tocar `configuration.yaml`.
  - **Estado** (`<prefijo>/<ciudad>/<estación>/state`): el JSON con el índice
    actual, la categoría y el máximo previsto a 24 h.

Uso:
    python mqtt_publish.py --host 192.168.1.50 --ciudad madrid
    python mqtt_publish.py --host broker --port 8883 --tls --user ha --password ***

    # Sin publicar: enseña por pantalla lo que enviaría
    python mqtt_publish.py --host x --dry-run

Requiere `paho-mqtt` (dependencia opcional: sólo la necesita este script).
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any

DEFAULT_PREFIX = "calidad_aire"
DISCOVERY_PREFIX = "homeassistant"


def _slug(text: str) -> str:
    """Identificador seguro para topics y object_id de Home Assistant."""
    out = []
    for ch in str(text).lower():
        if ch.isalnum():
            out.append(ch)
        elif ch in " -_/.":
            out.append("_")
        # el resto (acentos, símbolos) se descarta: los topics deben ser ASCII
    return "".join(out).strip("_") or "estacion"


def build_discovery(estacion: dict, ciudad: str, prefix: str) -> tuple[str, dict]:
    """(topic, payload) de la config de autodescubrimiento de una estación.

    `device` agrupa todas las estaciones de una ciudad bajo un mismo dispositivo
    en la UI de Home Assistant, en vez de dejarlas sueltas.
    """
    est_id = _slug(estacion.get("id") or estacion.get("codigo") or estacion.get("nombre"))
    ciudad_id = _slug(ciudad)
    object_id = f"{ciudad_id}_{est_id}"
    state_topic = f"{prefix}/{ciudad_id}/{est_id}/state"

    payload = {
        "name": estacion.get("nombre") or est_id,
        "unique_id": f"calidad_aire_{object_id}",
        "state_topic": state_topic,
        "value_template": "{{ value_json.ica_actual }}",
        "json_attributes_topic": state_topic,
        "unit_of_measurement": "ICA",
        "state_class": "measurement",
        "icon": "mdi:air-filter",
        "device": {
            "identifiers": [f"calidad_aire_{ciudad_id}"],
            "name": f"Calidad del aire — {ciudad.title()}",
            "manufacturer": "prediccion-calidad-aire",
            "model": "DCRNN-lite",
        },
    }
    topic = f"{DISCOVERY_PREFIX}/sensor/{prefix}_{object_id}/config"
    return topic, payload


def build_state(estacion: dict, ciudad: str, prefix: str) -> tuple[str, dict]:
    """(topic, payload) del estado de una estación."""
    est_id = _slug(estacion.get("id") or estacion.get("codigo") or estacion.get("nombre"))
    ciudad_id = _slug(ciudad)

    pred = estacion.get("ica_prediccion_24h") or []
    payload: dict[str, Any] = {
        "ica_actual": estacion.get("ica_actual"),
        "categoria": estacion.get("ica_categoria"),
        "estacion": estacion.get("nombre"),
        "latitude": estacion.get("lat"),
        "longitude": estacion.get("lon"),
    }
    if pred:
        payload["ica_max_24h"] = round(max(pred), 1)
        payload["horas_previstas"] = len(pred)
    return f"{prefix}/{ciudad_id}/{est_id}/state", payload


def fetch_estaciones(api: str, ciudad: str) -> list[dict]:
    import urllib.request

    url = f"{api.rstrip('/')}/ml/calidad-aire/estaciones?ciudad={ciudad}"
    with urllib.request.urlopen(url, timeout=30) as r:
        data = json.loads(r.read().decode())
    if not data.get("ok", True):
        raise RuntimeError(f"La API devolvió error: {data.get('error')}")
    return data.get("estaciones", [])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", required=True, help="Host del broker MQTT")
    ap.add_argument("--port", type=int, default=1883)
    ap.add_argument("--user")
    ap.add_argument("--password")
    ap.add_argument("--tls", action="store_true", help="Conectar con TLS")
    ap.add_argument("--ciudad", default="madrid")
    ap.add_argument("--prefix", default=DEFAULT_PREFIX, help="Prefijo de los topics de estado")
    ap.add_argument("--api", default="http://127.0.0.1:8091", help="Base de la API de calidad-aire")
    ap.add_argument("--no-discovery", action="store_true", help="No publicar la config de descubrimiento")
    ap.add_argument("--dry-run", action="store_true", help="Mostrar lo que se publicaría, sin conectar")
    args = ap.parse_args()

    try:
        estaciones = fetch_estaciones(args.api, args.ciudad)
    except Exception as e:
        print(f"error: no se pudo consultar la API ({e})", file=sys.stderr)
        return 1
    if not estaciones:
        print("error: la API no devolvió estaciones", file=sys.stderr)
        return 1

    mensajes: list[tuple[str, dict, bool]] = []  # (topic, payload, retain)
    for est in estaciones:
        if not args.no_discovery:
            t, p = build_discovery(est, args.ciudad, args.prefix)
            mensajes.append((t, p, True))   # retained: HA lo recupera al reiniciar
        t, p = build_state(est, args.ciudad, args.prefix)
        mensajes.append((t, p, False))

    if args.dry_run:
        for topic, payload, retain in mensajes:
            print(f"{'[retained] ' if retain else '           '}{topic}")
            print(f"            {json.dumps(payload, ensure_ascii=False)}")
        print(f"\n{len(mensajes)} mensajes ({len(estaciones)} estaciones) — dry-run, nada publicado")
        return 0

    try:
        import paho.mqtt.client as mqtt
    except ImportError:
        print("error: falta paho-mqtt (pip install paho-mqtt)", file=sys.stderr)
        return 1

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    if args.user:
        client.username_pw_set(args.user, args.password or "")
    if args.tls:
        client.tls_set()

    try:
        client.connect(args.host, args.port, keepalive=30)
    except Exception as e:
        print(f"error: no se pudo conectar a {args.host}:{args.port} ({e})", file=sys.stderr)
        return 1

    client.loop_start()
    try:
        for topic, payload, retain in mensajes:
            info = client.publish(topic, json.dumps(payload, ensure_ascii=False), qos=1, retain=retain)
            info.wait_for_publish(timeout=10)
    finally:
        client.loop_stop()
        client.disconnect()

    print(f"publicados {len(mensajes)} mensajes de {len(estaciones)} estaciones en {args.host}:{args.port}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
