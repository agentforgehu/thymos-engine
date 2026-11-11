"""Reproducible experiments for Thymos.

Run with:  python experiments/run_experiments.py

Outputs figures to docs/figures and a machine-readable summary to
docs/experiments.json.  The console prints a Markdown table used in
docs/EXPERIMENTS.md.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from thymos import Profile, Thymos, ThymosConfig
from thymos.endocrine import EndocrineSystem, circadian_factors
from thymos.forgetting import (
    EBBINGHAUS_SAVINGS,
    curve_factor,
    initial_stability,
    retrievability,
    update_stability,
)

FIG_DIR = os.path.join(ROOT, "docs", "figures")
DOCS = os.path.join(ROOT, "docs")

INK = "#1f2937"
MUTED = "#6b7280"
GRID = "#e5e7eb"
ACCENT = "#0f766e"
ACCENT_2 = "#7c3aed"
WARM = "#ea580c"
ROSE = "#e11d48"
BLUE = "#2563eb"


def setup_style() -> None:
    for path in (
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ):
        if os.path.exists(path):
            try:
                font_manager.fontManager.addfont(path)
            except Exception:
                pass
    plt.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": GRID,
            "axes.labelcolor": MUTED,
            "axes.titlecolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "font.size": 10,
        }
    )


def save(fig, name: str) -> str:
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 1. Three forgetting functions against Ebbinghaus' savings data
# ---------------------------------------------------------------------------
def exp_retention_models() -> dict:
    hours = np.array([h for h, _ in EBBINGHAUS_SAVINGS], dtype=float)
    y = np.array([v / 100.0 for _, v in EBBINGHAUS_SAVINGS], dtype=float)
    days = hours / 24.0

    def rmse(pred):
        return float(np.sqrt(np.mean((pred - y) ** 2)))

    # exponential with asymptote
    best_exp = (1e9, None)
    for c in np.arange(0.0, 0.31, 0.02):
        for tau in np.geomspace(0.1, 200.0, 240):
            pred = c + (1 - c) * np.exp(-hours / tau)
            e = rmse(pred)
            if e < best_exp[0]:
                best_exp = (e, (c, tau))

    # stretched exponential with asymptote
    best_str = (1e9, None)
    for c in np.arange(0.0, 0.31, 0.02):
        for tau in np.geomspace(0.2, 300.0, 80):
            for beta in np.arange(0.1, 0.95, 0.05):
                pred = c + (1 - c) * np.exp(-((hours / tau) ** beta))
                e = rmse(pred)
                if e < best_str[0]:
                    best_str = (e, (c, tau, beta))

    # FSRS power law (no asymptote)
    f = curve_factor(-0.5)
    best_fsrs = (1e9, None)
    for s in np.geomspace(0.01, 500.0, 400):
        pred = (1 + f * days / s) ** -0.5
        e = rmse(pred)
        if e < best_fsrs[0]:
            best_fsrs = (e, s)

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.scatter(hours, y, s=46, color=INK, zorder=5, label="Ebbinghaus 节省量数据")
    grid = np.geomspace(0.2, 800, 500)
    c, tau = best_exp[1]
    ax.plot(grid, c + (1 - c) * np.exp(-grid / tau), color=BLUE, lw=2,
            label=f"指数+渐近线 (RMSE={best_exp[0]:.3f})")
    c2, tau2, beta = best_str[1]
    ax.plot(grid, c2 + (1 - c2) * np.exp(-((grid / tau2) ** beta)), color=ACCENT, lw=2,
            label=f"拉伸指数 (RMSE={best_str[0]:.3f})")
    ax.plot(grid, (1 + f * (grid / 24) / best_fsrs[1]) ** -0.5, color=ROSE, lw=2,
            label=f"FSRS 幂律 (RMSE={best_fsrs[0]:.3f})")
    ax.set_xscale("log")
    ax.set_xlabel("距初次学习的时间（小时，对数轴）")
    ax.set_ylabel("保持率 / 节省量")
    ax.set_ylim(0, 1.02)
    ax.set_title("三种遗忘函数对 Ebbinghaus 数据的拟合", loc="left", fontsize=12, fontweight="bold")
    ax.grid(True, which="both", alpha=0.55)
    ax.legend(frameon=False, fontsize=8.6, loc="upper right")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    save(fig, "fig1_retention_models.png")

    return {
        "data_points": len(y),
        "exponential": {"rmse": round(best_exp[0], 4), "asymptote": round(best_exp[1][0], 3), "tau_hours": round(best_exp[1][1], 2)},
        "stretched": {"rmse": round(best_str[0], 4), "asymptote": round(best_str[1][0], 3), "tau_hours": round(best_str[1][1], 2), "beta": round(best_str[1][2], 3)},
        "fsrs": {"rmse": round(best_fsrs[0], 4), "stability_days": round(best_fsrs[1], 4)},
    }


# ---------------------------------------------------------------------------
# 2. Massed vs spaced review
# ---------------------------------------------------------------------------
def exp_spacing() -> dict:
    s0 = initial_stability(0.6, 0.2, 0.3, 1.0, base_days=0.5)

    def schedule(review_days, quality=0.85):
        s = s0
        last = 0.0
        for day in review_days:
            gap = day - last
            r = retrievability(gap, s)
            s = update_stability(s, r, quality=quality)
            last = day
        return s, last

    massed_s, massed_last = schedule([0.0, 0.02, 0.04, 0.06, 0.08])
    spaced_s, spaced_last = schedule([0.0, 1.0, 3.0, 7.0, 14.0])

    days = np.linspace(0, 60, 600)
    massed_r = np.array([retrievability(max(0.0, d - massed_last), massed_s) for d in days])
    spaced_r = np.array([retrievability(max(0.0, d - spaced_last), spaced_s) for d in days])

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.plot(days, massed_r, color=WARM, lw=2.2, label=f"集中复习 5 次（S={massed_s:.2f} 天）")
    ax.plot(days, spaced_r, color=ACCENT, lw=2.2, label=f"间隔复习 5 次（S={spaced_s:.2f} 天）")
    ax.axhline(0.9, color=MUTED, ls=":", lw=1)
    ax.text(59, 0.905, "90% 保持", ha="right", va="bottom", fontsize=8, color=MUTED)
    ax.set_xlabel("天数")
    ax.set_ylabel("可提取度 R(t, S)")
    ax.set_ylim(0, 1.02)
    ax.set_title("间隔效应：同样的复习次数，分布不同则长期保持不同", loc="left", fontsize=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, alpha=0.55)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    save(fig, "fig2_spacing.png")

    return {
        "massed_stability_days": round(massed_s, 3),
        "spaced_stability_days": round(spaced_s, 3),
        "ratio": round(spaced_s / massed_s, 2),
        "retention_day30_massed": round(float(retrievability(max(0, 30 - massed_last), massed_s)), 3),
        "retention_day30_spaced": round(float(retrievability(max(0, 30 - spaced_last), spaced_s)), 3),
    }


# ---------------------------------------------------------------------------
# 3. Acute stress: cortisol delay and retrieval impairment
# ---------------------------------------------------------------------------
def exp_stress() -> dict:
    endo = EndocrineSystem()
    circ = circadian_factors(10.0)
    minutes = np.arange(0, 241, 5)
    cortisol, excess, retrieval = [], [], []
    stressed = False
    stress_until = 60
    for m in minutes:
        stress = 1.0 if (30 <= m < stress_until) else 0.0
        if m == 30:
            stressed = True
        endo.step(5, stress=stress, circadian=circ)
        cortisol.append(endo.value("cortisol"))
        excess.append(endo.cortisol_excess())
        retrieval.append(endo.retrieval_gain())

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.plot(minutes, cortisol, color=ROSE, lw=2.2, label="皮质醇（相对单位）")
    ax.plot(minutes, excess, color=WARM, lw=1.8, ls="--", label="皮质醇超额（扣除昼夜预期）")
    ax.axvspan(30, stress_until, color="#fee2e2", alpha=0.6, lw=0)
    ax.text(31, 0.02, "30 分钟应激", fontsize=8.5, color=ROSE)
    ax.set_xlabel("分钟")
    ax.set_ylabel("相对水平")
    ax.set_ylim(0, 1.0)
    ax2 = ax.twinx()
    ax2.plot(minutes, retrieval, color=ACCENT, lw=2.0, label="提取增益")
    ax2.set_ylabel("提取增益", color=ACCENT)
    ax2.set_ylim(0.5, 1.3)
    ax2.tick_params(axis="y", colors=ACCENT)
    lines = ax.get_lines()[:2] + ax2.get_lines()[:1]
    ax.legend(lines, [l.get_label() for l in lines], frameon=False, fontsize=8.6, loc="upper left")
    ax.set_title("应激后的皮质醇延迟峰值会压低记忆提取增益", loc="left", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.5)
    for spine in ("top",):
        ax.spines[spine].set_visible(False)
    save(fig, "fig3_stress_memory.png")

    neutral_s = initial_stability(0.4, 0.0, 0.1, 1.0)
    emotional_s = initial_stability(0.8, 0.9, 0.85, 1.6)
    return {
        "peak_cortisol": round(float(max(cortisol)), 3),
        "peak_minute": int(minutes[int(np.argmax(cortisol))]),
        "min_retrieval_gain": round(float(min(retrieval)), 3),
        "neutral_stability_days": round(neutral_s, 3),
        "emotional_stability_days": round(emotional_s, 3),
        "emotional_advantage": round(emotional_s / neutral_s, 2),
    }


# ---------------------------------------------------------------------------
# 4. Weather coupling over 30 days
# ---------------------------------------------------------------------------
def exp_weather() -> dict:
    start = dt.datetime(2026, 3, 1, 8, 0, tzinfo=dt.timezone(dt.timedelta(hours=8))).timestamp()
    profile = Profile(name="天气对照", location_name="上海", latitude=31.23, longitude=121.47, timezone="Asia/Shanghai")

    def run(sensitivity: float):
        p = Profile(**{**profile.to_dict(), "weather_sensitivity": sensitivity})
        mind = Thymos(p, ThymosConfig(offline=True, seed=5), now=start)
        moods, temps, buckets = [], [], []
        for day in range(30):
            if day % 4 == 0:
                mind.perceive("今天完成了一件重要的事，感觉不错", importance=0.65, tags=["工作"])
            elif day % 4 == 2:
                mind.perceive("有点累，事情没有按计划推进", importance=0.5, tags=["工作"])
            mind.tick(24 * 3600)
            moods.append(mind.affect.state.mood_valence)
            temps.append(mind.environment.temp_c)
            buckets.append(mind.environment.weather_bucket)
        return np.array(moods), np.array(temps), buckets

    base, temps, buckets = run(0.0)
    sensitive, _, _ = run(1.5)
    diff = sensitive - base

    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    xs = np.arange(1, 31)
    ax.plot(xs, base, color=MUTED, lw=1.8, label="天气敏感度 0（对照）")
    ax.plot(xs, sensitive, color=ACCENT, lw=2.0, label="天气敏感度 1.5")
    ax.fill_between(xs, base, sensitive, color=ACCENT, alpha=0.12)
    ax.set_xlabel("天数")
    ax.set_ylabel("心境值")
    ax.set_title("天气耦合对 30 天心境轨迹的影响（上海，合成天气）", loc="left", fontsize=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, alpha=0.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    save(fig, "fig4_weather_mood.png")

    rainy = sum(1 for b in buckets if b == "rain")
    return {
        "mean_abs_difference": round(float(np.mean(np.abs(diff))), 4),
        "max_abs_difference": round(float(np.max(np.abs(diff))), 4),
        "corr_with_temperature": round(float(np.corrcoef(sensitive, temps)[0, 1]), 3),
        "rainy_days": rainy,
        "final_mood_base": round(float(base[-1]), 3),
        "final_mood_sensitive": round(float(sensitive[-1]), 3),
    }


# ---------------------------------------------------------------------------
# 5. Emotional vs neutral forgetting
# ---------------------------------------------------------------------------
def exp_emotional_forgetting() -> dict:
    neutral = initial_stability(0.4, 0.0, 0.1, 1.0)
    mild = initial_stability(0.6, 0.5, 0.5, 1.1)
    intense = initial_stability(0.8, 0.9, 0.85, 1.6)
    days = np.linspace(0, 60, 500)

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    for s, label, color in ((neutral, "中性事件", MUTED), (mild, "轻度情绪", BLUE), (intense, "高唤醒情绪", ROSE)):
        ax.plot(days, [retrievability(d, s) for d in days], lw=2.1, color=color,
                label=f"{label}（S={s:.2f} 天）")
    ax.axhline(0.5, color=GRID, ls="--", lw=1)
    ax.set_xlabel("天数")
    ax.set_ylabel("可提取度")
    ax.set_ylim(0, 1.02)
    ax.set_title("情绪唤醒提高初始稳定度，但不会让记忆永不遗忘", loc="left", fontsize=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, alpha=0.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    save(fig, "fig5_emotional_forgetting.png")

    def half_life(s):
        lo, hi = 0.0, 5000.0
        for _ in range(80):
            mid = (lo + hi) / 2
            if retrievability(mid, s) > 0.5:
                lo = mid
            else:
                hi = mid
        return lo

    return {
        "neutral_half_life_days": round(half_life(neutral), 2),
        "mild_half_life_days": round(half_life(mild), 2),
        "intense_half_life_days": round(half_life(intense), 2),
    }


def main() -> int:
    setup_style()
    results = {
        "retention_models": exp_retention_models(),
        "spacing": exp_spacing(),
        "stress": exp_stress(),
        "weather": exp_weather(),
        "emotional_forgetting": exp_emotional_forgetting(),
    }
    os.makedirs(DOCS, exist_ok=True)
    with open(os.path.join(DOCS, "experiments.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)

    print("| 实验 | 关键指标 | 结果 |")
    print("|---|---|---|")
    r = results["retention_models"]
    print(f"| 遗忘函数拟合 | RMSE（指数 / 拉伸 / FSRS） | {r['exponential']['rmse']} / {r['stretched']['rmse']} / {r['fsrs']['rmse']} |")
    s = results["spacing"]
    print(f"| 间隔效应 | 30 天保持率（集中 vs 间隔） | {s['retention_day30_massed']:.0%} vs {s['retention_day30_spaced']:.0%} |")
    st = results["stress"]
    print(f"| 应激 | 皮质醇峰值分钟 / 最低提取增益 | {st['peak_minute']} / {st['min_retrieval_gain']} |")
    w = results["weather"]
    print(f"| 天气耦合 | 平均绝对心境差 / 最大差 | {w['mean_abs_difference']} / {w['max_abs_difference']} |")
    e = results["emotional_forgetting"]
    print(f"| 情绪记忆 | 半衰期（中性/轻度/强烈，天） | {e['neutral_half_life_days']} / {e['mild_half_life_days']} / {e['intense_half_life_days']} |")
    print(f"\n图表已写入 {FIG_DIR}")
    print(f"指标已写入 {os.path.join(DOCS, 'experiments.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
