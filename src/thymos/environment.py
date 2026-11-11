"""Place, daylight, season and weather as small, bounded modulators.

The literature reviewed in docs/RESEARCH.md supports real but small average
effects of weather on mood, with large individual differences.  Nothing here
should be read as a deterministic rule such as "rain makes people sad"; the
effects are deliberately bounded and multiplied by a per-profile sensitivity.
"""

from __future__ import annotations

import datetime as dt
import math
from typing import Any

from .affect import EnvironmentEffects
from .config import Profile
from .mathx import clamp
from .types import EnvironmentState

try:  # Python 3.9+
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore


def local_datetime(timestamp: float, timezone: str) -> dt.datetime:
    if ZoneInfo is not None:
        try:
            return dt.datetime.fromtimestamp(timestamp, tz=ZoneInfo(timezone))
        except Exception:
            pass
    return dt.datetime.fromtimestamp(timestamp, tz=dt.timezone.utc)


def solar_declination(day_of_year: int) -> float:
    """Solar declination in degrees (Cooper's approximation)."""
    return 23.44 * math.sin(2 * math.pi * (284 + day_of_year) / 365.0)


def daylight_hours(latitude: float, day_of_year: int) -> float:
    decl = math.radians(solar_declination(day_of_year))
    lat = math.radians(clamp(latitude, -66.0, 66.0))
    cos_h = -math.tan(lat) * math.tan(decl)
    cos_h = clamp(cos_h, -1.0, 1.0)
    return 24.0 * math.acos(cos_h) / math.pi


def season_for(latitude: float, month: int) -> str:
    north = latitude >= 0
    if month in (3, 4, 5):
        return "spring" if north else "autumn"
    if month in (6, 7, 8):
        return "summer" if north else "winter"
    if month in (9, 10, 11):
        return "autumn" if north else "spring"
    return "winter" if north else "summer"


def weather_bucket(temp_c: float, humidity: float, cloud: float, precipitation: float) -> str:
    if precipitation >= 0.2:
        return "rain"
    if temp_c >= 29.0:
        return "hot"
    if temp_c <= 5.0:
        return "cold"
    if humidity >= 78.0 and temp_c >= 24.0:
        return "humid"
    if cloud <= 30.0:
        return "clear"
    if cloud >= 75.0:
        return "overcast"
    return "mild"


def build_environment(
    profile: Profile,
    timestamp: float,
    *,
    temp_c: float,
    apparent_c: float | None = None,
    humidity: float = 60.0,
    pressure_hpa: float = 1013.0,
    cloud_cover: float = 40.0,
    precipitation_mm: float = 0.0,
    radiation_wm2: float = 300.0,
    wind_kph: float = 8.0,
    condition: str = "clear",
    weather_code: int = 0,
    source: str = "synthetic",
    previous_pressure: float | None = None,
) -> EnvironmentState:
    local = local_datetime(timestamp, profile.timezone)
    doy = int(local.strftime("%j"))
    daylight = daylight_hours(profile.latitude, doy)
    sunrise = 12.0 - daylight / 2.0
    sunset = 12.0 + daylight / 2.0
    hour = local.hour + local.minute / 60.0
    is_day = sunrise <= hour <= sunset
    if radiation_wm2 <= 0 and is_day:
        radiation_wm2 = max(0.0, 780.0 * math.sin(math.pi * (hour - sunrise) / max(0.1, daylight)) * (1.0 - cloud_cover / 130.0))
    if apparent_c is None:
        apparent_c = temp_c
    trend = 0.0 if previous_pressure is None else pressure_hpa - previous_pressure

    return EnvironmentState(
        timestamp=timestamp,
        location_name=profile.location_name,
        latitude=profile.latitude,
        longitude=profile.longitude,
        timezone=profile.timezone,
        temp_c=round(float(temp_c), 2),
        apparent_c=round(float(apparent_c), 2),
        humidity=round(float(humidity), 1),
        pressure_hpa=round(float(pressure_hpa), 1),
        cloud_cover=round(float(cloud_cover), 1),
        precipitation_mm=round(float(precipitation_mm), 2),
        radiation_wm2=round(float(radiation_wm2), 1),
        wind_kph=round(float(wind_kph), 1),
        condition=condition,
        weather_code=int(weather_code),
        daylight_hours=round(daylight, 3),
        is_day=bool(is_day),
        season=season_for(profile.latitude, local.month),
        weather_bucket=weather_bucket(temp_c, humidity, cloud_cover, precipitation_mm),
        source=source,
        fetched_at=timestamp,
        pressure_trend_3h=round(trend, 2),
    )


def comfort_temperature(profile: Profile) -> float:
    """Adapted neutral temperature: warmer climates accept warmer comfort."""
    extra = max(0.0, (abs(profile.latitude) - 35.0)) * 0.08
    return 22.0 + profile.comfort_offset_c + min(2.0, extra)


def environment_effects(state: EnvironmentState, profile: Profile) -> EnvironmentEffects:
    """Map weather to a *small* mood/arousal bias plus endocrine drives."""
    sensitivity = max(0.0, profile.weather_sensitivity)
    light_sensitivity = max(0.0, profile.light_sensitivity)

    cloud = clamp(state.cloud_cover / 100.0)
    sunshine = clamp(state.radiation_wm2 / 650.0) if state.radiation_wm2 > 0 else 1.0 - cloud
    comfort = comfort_temperature(profile)
    deviation = abs(state.apparent_c - comfort)
    temp_stress = clamp((deviation - 4.0) / 12.0)
    rain = clamp(state.precipitation_mm / 4.0)
    pressure_drop = clamp(-state.pressure_trend_3h / 8.0)
    daylight_anomaly = clamp((state.daylight_hours - 12.0) / 6.0, -1.0, 1.0)

    mood = sensitivity * (
        0.070 * sunshine * light_sensitivity
        - 0.050 * temp_stress
        - 0.035 * rain
        - 0.030 * pressure_drop
        + 0.025 * daylight_anomaly * light_sensitivity
    )
    arousal = sensitivity * (0.050 * temp_stress + 0.020 * rain - 0.030 * pressure_drop)
    mood = clamp(mood, -0.12, 0.12)
    arousal = clamp(arousal, -0.08, 0.08)

    return EnvironmentEffects(
        mood_delta=mood,
        arousal_delta=arousal,
        serotonin_drive=clamp(0.35 + 0.50 * sunshine + 0.15 * daylight_anomaly),
        melatonin_drive=clamp(0.20 + 0.60 * (1.0 - sunshine) - 0.20 * clamp(state.radiation_wm2 / 500.0)),
        energy=clamp(0.55 + 0.30 * sunshine - 0.30 * temp_stress - 0.20 * rain),
    )


def sleep_debt(profile: Profile, hour: float, average_sleep: float | None = None) -> float:
    """Coarse sleep pressure in [0, 1] from time since waking and sleep length."""
    sleep_len = average_sleep if average_sleep is not None else profile.sleep_hours
    deficit = clamp((7.5 - sleep_len) / 3.0, 0.0, 1.0)
    wake = profile.wake_hour % 24.0
    awake_hours = (hour - wake) % 24.0
    if awake_hours > 18.0:
        awake_hours = 18.0
    pressure = clamp(awake_hours / 18.0)
    return clamp(0.55 * deficit + 0.45 * pressure)


def snapshot(state: EnvironmentState, effects: EnvironmentEffects | None = None) -> dict[str, Any]:
    data = state.to_dict()
    if effects is not None:
        data["effects"] = effects.to_dict()
    return data
