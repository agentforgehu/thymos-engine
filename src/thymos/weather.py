"""Weather providers: Open-Meteo when online, deterministic synthetic otherwise.

Open-Meteo is free for non-commercial use and needs no API key.  Requests are
cached to disk because both rate limits and privacy benefit from fewer calls.
Only a coarse location is sent; no user identifier is included.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import time
import urllib.parse
import urllib.request
from typing import Any

from .config import Profile, ThymosConfig
from .environment import build_environment, daylight_hours, local_datetime
from .types import EnvironmentState

WMO = {
    0: "clear", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "depositing rime fog", 51: "light drizzle", 53: "drizzle",
    55: "dense drizzle", 56: "freezing drizzle", 57: "dense freezing drizzle",
    61: "slight rain", 63: "rain", 65: "heavy rain", 66: "freezing rain",
    67: "heavy freezing rain", 71: "slight snow", 73: "snow", 75: "heavy snow",
    77: "snow grains", 80: "slight showers", 81: "showers", 82: "violent showers",
    85: "snow showers", 86: "heavy snow showers", 95: "thunderstorm",
    96: "thunderstorm with hail", 99: "thunderstorm with heavy hail",
}


class WeatherClient:
    def __init__(self, config: ThymosConfig, cache_path: str | None = None) -> None:
        self.config = config
        self.cache_path = cache_path or config.environment_cache_path
        self._memory_cache: dict[str, Any] = {}

    # ----------------------------------------------------------------- public
    def current(self, profile: Profile, timestamp: float | None = None) -> EnvironmentState:
        ts = time.time() if timestamp is None else timestamp
        if self.config.offline:
            return self.synthetic(profile, ts)
        key = self._key(profile)
        cached = self._read_cache(key)
        if cached and abs(ts - float(cached.get("fetched_at", 0))) <= self.config.weather_ttl_seconds:
            return self._from_cache(profile, cached, ts)

        try:
            payload = self._fetch(profile)
            normalized = self._normalize(payload)
            normalized["fetched_at"] = ts
            normalized["source"] = "open-meteo"
            self._write_cache(key, normalized)
            build_args = {k: v for k, v in normalized.items() if k not in {"fetched_at", "source", "daylight_hours"}}
            return build_environment(
                profile,
                ts,
                source="open-meteo",
                previous_pressure=cached.get("pressure_hpa") if cached else None,
                **build_args,
            )
        except Exception:
            if cached:
                return self._from_cache(profile, cached, ts, source="cache")
            return self.synthetic(profile, ts)

    def synthetic(self, profile: Profile, timestamp: float) -> EnvironmentState:
        """Deterministic plausible weather, used offline and in tests."""
        local = local_datetime(timestamp, profile.timezone)
        day = local.timetuple().tm_yday
        hour = local.hour + local.minute / 60.0
        doy_key = int(local.strftime("%Y%m%d"))
        seed = int(hashlib.blake2b(f"{profile.latitude:.2f},{profile.longitude:.2f},{doy_key}".encode(), digest_size=8).hexdigest(), 16)
        noise = ((seed % 1000) / 1000.0 - 0.5) * 2.0

        seasonal = math.cos(2 * math.pi * (day - 200) / 365.0)  # northern-hemisphere summer peak
        if profile.latitude < 0:
            seasonal = -seasonal
        base = 18.0 + 10.0 * seasonal * min(1.0, abs(profile.latitude) / 45.0) + 3.0 * noise
        diurnal = 4.5 * math.sin(2 * math.pi * (hour - 9.0) / 24.0)
        temp = base + diurnal

        cloud = 35.0 + 45.0 * ((seed >> 8) % 1000) / 1000.0
        rain = 0.0
        if cloud > 70 and ((seed >> 16) % 100) < 45:
            rain = 0.4 + 3.0 * ((seed >> 24) % 1000) / 1000.0
        humidity = 45.0 + 40.0 * (cloud / 100.0) + (8.0 if rain else 0.0)
        pressure = 1008.0 + 12.0 * math.sin(2 * math.pi * (day + 40) / 9.0) + 4.0 * noise
        wind = 4.0 + 12.0 * ((seed >> 32) % 1000) / 1000.0
        daylight = daylight_hours(profile.latitude, day)
        code = 61 if rain else (3 if cloud > 75 else 1 if cloud < 30 else 2)

        return build_environment(
            profile,
            timestamp,
            temp_c=temp,
            apparent_c=temp + (0.4 if humidity > 75 else 0.0) - (0.6 if wind > 15 else 0.0),
            humidity=humidity,
            pressure_hpa=pressure,
            cloud_cover=cloud,
            precipitation_mm=rain,
            radiation_wm2=0.0,
            wind_kph=wind,
            condition=WMO.get(code, "unknown"),
            weather_code=code,
            source="synthetic",
        )

    # ---------------------------------------------------------------- remote
    def _fetch(self, profile: Profile) -> dict[str, Any]:
        params = {
            "latitude": f"{profile.latitude:.2f}",
            "longitude": f"{profile.longitude:.2f}",
            "current": ",".join(
                [
                    "temperature_2m",
                    "relative_humidity_2m",
                    "apparent_temperature",
                    "precipitation",
                    "weather_code",
                    "cloud_cover",
                    "pressure_msl",
                    "wind_speed_10m",
                    "shortwave_radiation",
                ]
            ),
            "daily": "daylight_duration,sunrise,sunset",
            "timezone": "auto",
            "forecast_days": "1",
        }
        url = "https://api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode(params)
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "thymos-engine/0.1 (+https://github.com/agentforgehu/thymos-engine)"},
        )
        with urllib.request.urlopen(request, timeout=self.config.weather_timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))

    @staticmethod
    def _normalize(payload: dict[str, Any]) -> dict[str, Any]:
        current = payload.get("current") or {}
        daily = payload.get("daily") or {}
        code = int(current.get("weather_code", 0) or 0)
        daylight = 12.0
        if daily.get("daylight_duration"):
            daylight = float(daily["daylight_duration"][0]) / 3600.0
        return {
            "temp_c": float(current.get("temperature_2m", 20.0)),
            "apparent_c": float(current.get("apparent_temperature", current.get("temperature_2m", 20.0))),
            "humidity": float(current.get("relative_humidity_2m", 60.0)),
            "pressure_hpa": float(current.get("pressure_msl", 1013.0)),
            "cloud_cover": float(current.get("cloud_cover", 40.0)),
            "precipitation_mm": float(current.get("precipitation", 0.0)),
            "radiation_wm2": float(current.get("shortwave_radiation", 0.0) or 0.0),
            "wind_kph": float(current.get("wind_speed_10m", 8.0)),
            "condition": WMO.get(code, "unknown"),
            "weather_code": code,
            "daylight_hours": daylight,
        }

    # ----------------------------------------------------------------- cache
    def _key(self, profile: Profile) -> str:
        return f"{profile.latitude:.2f},{profile.longitude:.2f}"

    def _read_cache(self, key: str) -> dict[str, Any] | None:
        if key in self._memory_cache:
            return self._memory_cache[key]
        try:
            with open(self.cache_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            entry = data.get(key)
            if entry:
                self._memory_cache[key] = entry
                return entry
        except (OSError, ValueError):
            pass
        return None

    def _write_cache(self, key: str, value: dict[str, Any]) -> None:
        data: dict[str, Any] = {}
        try:
            with open(self.cache_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            data = {}
        data[key] = value
        os.makedirs(os.path.dirname(os.path.abspath(self.cache_path)), exist_ok=True)
        tmp = f"{self.cache_path}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, self.cache_path)
        self._memory_cache[key] = value

    def _from_cache(self, profile: Profile, entry: dict[str, Any], ts: float, source: str | None = None) -> EnvironmentState:
        data = {k: v for k, v in entry.items() if k not in {"fetched_at", "source", "daylight_hours"}}
        return build_environment(
            profile,
            ts,
            source=source or entry.get("source", "cache"),
            previous_pressure=entry.get("pressure_hpa"),
            **data,
        )


def geocode(name: str, timeout: float = 8.0) -> dict[str, Any] | None:
    """Resolve a city name through Open-Meteo's geocoding API (no key)."""
    params = {"name": name, "count": "1", "language": "zh", "format": "json"}
    url = "https://geocoding-api.open-meteo.com/v1/search?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(url, headers={"User-Agent": "thymos/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        results = payload.get("results") or []
        if not results:
            return None
        hit = results[0]
        return {
            "name": hit.get("name"),
            "latitude": hit.get("latitude"),
            "longitude": hit.get("longitude"),
            "timezone": hit.get("timezone", "UTC"),
            "country": hit.get("country"),
        }
    except Exception:
        return None
