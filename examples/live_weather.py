"""Live weather example (network required).

Run:
    python examples/live_weather.py
    python examples/live_weather.py 北京
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from thymos import Profile, Thymos, ThymosConfig
from thymos.weather import geocode


def main() -> None:
    city = sys.argv[1] if len(sys.argv) > 1 else "上海"
    hit = geocode(city)
    if hit:
        profile = Profile(
            name="旅行者",
            location_name=hit["name"] or city,
            latitude=hit["latitude"],
            longitude=hit["longitude"],
            timezone=hit.get("timezone") or "Asia/Shanghai",
        )
        print(f"定位：{profile.location_name} ({profile.latitude:.3f}, {profile.longitude:.3f}) {profile.timezone}")
    else:
        profile = Profile(location_name=city)
        print("地理编码失败，使用上海默认坐标。")

    mind = Thymos(profile, ThymosConfig(offline=False, weather_ttl_seconds=600))
    mind.refresh_environment(force=True)
    env = mind.environment
    effects = mind._effects
    snap = mind.snapshot()

    print(f"天气：{env.condition}，{env.temp_c:.1f}°C（体感 {env.apparent_c:.1f}°C）")
    print(f"湿度 {env.humidity:.0f}% / 气压 {env.pressure_hpa:.0f} hPa / 云量 {env.cloud_cover:.0f}%")
    print(f"昼长 {env.daylight_hours:.1f} h / 季节 {env.season} / 数据源 {env.source}")
    print(f"天气→心境偏置：{effects.mood_delta:+.3f}（被限制在 ±0.12）")
    print(f"当前心境：{snap['affect']['mood']['valence']:+.3f}，压力 {snap['affect']['stress']:.3f}")


if __name__ == "__main__":
    main()
