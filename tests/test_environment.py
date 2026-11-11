import datetime as dt

import pytest

from thymos.config import Profile
from thymos.environment import (
    build_environment,
    comfort_temperature,
    daylight_hours,
    environment_effects,
    season_for,
    sleep_debt,
    weather_bucket,
)
from thymos.weather import WeatherClient
from thymos.config import ThymosConfig


def test_equator_daylight_is_about_twelve_hours():
    for day in (80, 172, 266, 355):
        assert daylight_hours(0.0, day) == pytest.approx(12.0, abs=0.2)


def test_high_latitude_summer_days_are_longer():
    assert daylight_hours(60.0, 172) > daylight_hours(60.0, 355) + 6


def test_season_flips_across_hemispheres():
    assert season_for(40, 7) == "summer"
    assert season_for(-40, 7) == "winter"


def test_weather_buckets():
    assert weather_bucket(32, 50, 10, 0) == "hot"
    assert weather_bucket(2, 50, 10, 0) == "cold"
    assert weather_bucket(20, 90, 80, 1.5) == "rain"
    assert weather_bucket(20, 60, 10, 0) == "clear"


def test_comfort_temperature_adapts_to_latitude():
    temperate = comfort_temperature(Profile(latitude=35))
    tropical = comfort_temperature(Profile(latitude=5, comfort_offset_c=1.5))
    assert tropical > temperate


def test_environment_effects_are_bounded():
    profile = Profile(weather_sensitivity=3.0, light_sensitivity=3.0)
    ts = dt.datetime(2026, 7, 1, 14, 0, tzinfo=dt.timezone.utc).timestamp()
    env = build_environment(profile, ts, temp_c=38, humidity=95, cloud_cover=0, precipitation_mm=8, radiation_wm2=900)
    effects = environment_effects(env, profile)
    assert -0.12 <= effects.mood_delta <= 0.12
    assert -0.08 <= effects.arousal_delta <= 0.08
    assert 0.0 <= effects.energy <= 1.0


def test_sleep_debt_grows_late_in_the_day():
    profile = Profile(wake_hour=7.0, sleep_hours=7.5)
    assert sleep_debt(profile, 21.0) > sleep_debt(profile, 9.0)


def test_synthetic_weather_is_deterministic():
    client = WeatherClient(ThymosConfig(offline=True))
    profile = Profile()
    ts = 1770000000.0
    a = client.synthetic(profile, ts)
    b = client.synthetic(profile, ts)
    assert a.temp_c == b.temp_c
    assert a.weather_bucket == b.weather_bucket
    assert a.source == "synthetic"


def test_build_environment_marks_night_without_radiation():
    profile = Profile(latitude=31.23, timezone="Asia/Shanghai")
    night = dt.datetime(2026, 1, 15, 2, 0, tzinfo=dt.timezone(dt.timedelta(hours=8))).timestamp()
    env = build_environment(profile, night, temp_c=8, cloud_cover=20, radiation_wm2=0)
    assert env.is_day is False
    assert env.daylight_hours < 12
