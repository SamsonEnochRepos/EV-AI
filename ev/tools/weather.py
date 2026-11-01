"""Live weather from Open-Meteo (no account or API key).

Web-search snippets only describe weather sites, never the actual conditions, so weather gets its own tool. The
only thing sent is the place name (geocoding) and the resulting coordinates (forecast); nothing about the user.
The reply is built here from numeric fields and a fixed description table, so no external text reaches the model
except the place name, which is reduced to letters and punctuation first.
"""
from __future__ import annotations

import re
import threading
import time
from typing import Any

import httpx

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

_WMO = {
    0: "clear", 1: "mostly clear", 2: "partly cloudy", 3: "overcast", 45: "foggy", 48: "foggy",
    51: "light drizzle", 53: "drizzling", 55: "heavy drizzle", 56: "freezing drizzle", 57: "freezing drizzle",
    61: "lightly raining", 63: "raining", 65: "raining heavily", 66: "freezing rain", 67: "freezing rain",
    71: "lightly snowing", 73: "snowing", 75: "snowing heavily", 77: "snowing", 80: "showery",
    81: "showery", 82: "very showery", 85: "snow showers", 86: "snow showers",
    95: "thunderstorms", 96: "thunderstorms with hail", 99: "thunderstorms with hail",
}


# Reused connection (skips two TLS handshakes per question) and short caches. Only used when no client is passed in.
_GEO_TTL_S = 24 * 3600          # a city doesn't move
_FORECAST_TTL_S = 300
_shared: httpx.Client | None = None
_shared_lock = threading.Lock()
_geo: dict[str, tuple[float, dict[str, Any]]] = {}
_forecast: dict[tuple[float, float, str], tuple[float, dict[str, Any]]] = {}


def _shared_client(timeout: float) -> httpx.Client:
    global _shared
    with _shared_lock:
        if _shared is None:
            _shared = httpx.Client(timeout=timeout, follow_redirects=False)
        return _shared


class WeatherError(RuntimeError):
    """Message is safe to speak to the user."""


def _clean_place(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^A-Za-z .,'\-]", "", s)).strip()[:60]


def _num(v: Any) -> int:
    return int(round(float(v)))


def get_weather(place: str, day: str = "today", units: str = "metric", timeout: float = 8.0,
                http: httpx.Client | None = None) -> str:
    place = _clean_place(place)
    if not place:
        raise WeatherError("Which city do you want the weather for?")
    imperial = units == "imperial"
    cached = http is None
    client = http or _shared_client(timeout)
    t_now = time.monotonic()
    try:
        geo_key = place.split(",")[0].lower()
        hit = _geo.get(geo_key) if cached else None
        if hit and t_now - hit[0] < _GEO_TTL_S:
            loc = hit[1]
        else:
            g = client.get(GEOCODE_URL, params={"name": place.split(",")[0], "count": 1, "language": "en", "format": "json"},
                           timeout=timeout)
            g.raise_for_status()
            hits = g.json().get("results") or []
            if not hits:
                raise WeatherError(f"I couldn't find a place called {place}.")
            loc = hits[0]
            if cached:
                _geo[geo_key] = (t_now, loc)
        name = _clean_place(str(loc.get("name", place))) or place
        fkey = (float(loc["latitude"]), float(loc["longitude"]), units)
        fhit = _forecast.get(fkey) if cached else None
        if fhit and t_now - fhit[0] < _FORECAST_TTL_S:
            data = fhit[1]
        else:
            f = client.get(FORECAST_URL, params={
                "latitude": fkey[0], "longitude": fkey[1], "timezone": "auto",
                "current": "temperature_2m,apparent_temperature,weather_code",
                "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
                "forecast_days": 2,
                **({"temperature_unit": "fahrenheit"} if imperial else {})}, timeout=timeout)
            f.raise_for_status()
            data = f.json()
            if cached:
                _forecast[fkey] = (t_now, data)
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as e:
        raise WeatherError("I couldn't reach the weather service just now.") from e

    unit = "Fahrenheit" if imperial else "Celsius"
    try:
        d = data["daily"]
        i = 1 if day == "tomorrow" else 0
        hi, lo = _num(d["temperature_2m_max"][i]), _num(d["temperature_2m_min"][i])
        rain = d["precipitation_probability_max"][i]
        rain_s = f", with a {_num(rain)} percent chance of rain" if rain is not None else ""
        if day == "tomorrow":
            desc = _WMO.get(int(d["weather_code"][1]), "mixed")
            return f"Tomorrow in {name}: {desc}, a high of {hi} and a low of {lo} degrees {unit}{rain_s}."
        cur = data["current"]
        now = _WMO.get(int(cur["weather_code"]), "mixed")
        return (f"Right now in {name} it's {_num(cur['temperature_2m'])} degrees {unit} and {now}, feels like "
                f"{_num(cur['apparent_temperature'])}. Today's high is {hi} and the low is {lo}{rain_s}.")
    except (KeyError, IndexError, TypeError, ValueError) as e:
        raise WeatherError("The weather service sent something I couldn't read.") from e
