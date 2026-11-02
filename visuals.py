"""Interactive visualizations and a downloadable PNG report."""
from __future__ import annotations

from io import BytesIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from audio_engine import BANDS

STEM_ENGLISH = {"보컬": "Vocals", "드럼": "Drums", "베이스": "Bass", "기타 악기(묶음)": "Other", "기타(실험)": "Guitar", "피아노(실험)": "Piano"}


def overview_plot(result: dict, selected: dict) -> go.Figure:
    sp = result["spec"]
    t, f = sp["times"] + result["offset"], sp["freqs"]
    wave = sp["wave"]
    x = np.arange(len(wave)) / sp["sr"] + result["offset"]
    decimate = max(1, len(wave) // 4500)
    keep_freq = (f <= 12000)
    # modest resolution for responsive browser zoom
    stride_f = max(1, int(np.sum(keep_freq)//350))
    stride_t = max(1, len(t)//650)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.22, 0.58, 0.20], vertical_spacing=.08,
                        subplot_titles=("파형 / Waveform", "STFT 스펙트로그램", "특성 변화 점수 (검토 우선순위)"))
    fig.add_trace(go.Scatter(x=x[::decimate], y=wave[::decimate],
                             name="파형", line=dict(color="#4fb2c0", width=1)), row=1, col=1)
    fig.add_trace(go.Heatmap(x=t[::stride_t], y=f[keep_freq][::stride_f],
                             z=sp["db"][keep_freq, :][::stride_f, ::stride_t],
                             zmin=-75, zmax=0, colorscale="Magma", name="dB",
                             colorbar=dict(title="dB", len=.52, y=.52)), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=result["score"],
                             name="검토 점수", fill="tozeroy",
                             line=dict(color="#e4a33e", width=1.3)), row=3, col=1)
    for row in (1, 2, 3):
        fig.add_vrect(x0=selected["start"], x1=selected["end"],
                      line_width=1, line_color="#f37272", fillcolor="#f37272", opacity=.20,
                      row=row, col=1)
    fig.update_yaxes(title_text="진폭", row=1, col=1)
    fig.update_yaxes(title_text="Hz", row=2, col=1)
    fig.update_yaxes(title_text="변화 점수", row=3, col=1)
    fig.update_xaxes(title_text="원본 시간 (초)", row=3, col=1)
    fig.update_layout(height=690, margin=dict(t=55, l=35, b=35, r=35),
                      showlegend=False, hovermode="x unified")
    return fig


def stem_heatmap(records: list[dict]) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        x=list(BANDS.keys()),
        y=[r["stem"] for r in records],
        z=[[r[key] for key in BANDS] for r in records],
        zmin=-12, zmax=12, zmid=0,
        colorscale="RdBu_r", texttemplate="%{z:+.1f}",
        hovertemplate="%{y}<br>%{x}<br>다른 구간 대비 %{z:+.1f} dB<extra></extra>",
        colorbar=dict(title="상대 dB"),
    ))
    fig.update_layout(title="선택 구간에서 증가/감소한 대역 에너지 (왜곡 확정 아님)",
                      height=360, margin=dict(t=55, b=65, l=35, r=25),
                      xaxis=dict(tickangle=-15), yaxis=dict(autorange="reversed"))
    return fig


def export_report(result: dict, selected: dict, records: list[dict] | None = None) -> bytes:
    sp = result["spec"]
    f, t, wave = sp["freqs"], sp["times"] + result["offset"], sp["wave"]
    if records:
        fig, axes = plt.subplots(4, 1, figsize=(12, 12),
                                 gridspec_kw={"height_ratios": [1, 2, .75, 1.4]},
                                 constrained_layout=True)
    else:
        fig, axes = plt.subplots(3, 1, figsize=(12, 9),
                                 gridspec_kw={"height_ratios": [1, 2, .8]},
                                 constrained_layout=True)
    i = max(1, len(wave)//4000)
    axes[0].plot(np.arange(0, len(wave), i)/sp["sr"] + result["offset"], wave[::i], lw=.8)
    axes[0].set(title="Waveform", ylabel="Amplitude")
    sel = f <= 12000
    axes[1].pcolormesh(t, f[sel], sp["db"][sel], shading="auto", cmap="magma", vmin=-75, vmax=0)
    axes[1].set(title="STFT spectrogram", ylabel="Frequency (Hz)")
    axes[2].plot(t, result["score"], lw=1, color="#bc780c")
    axes[2].set(title="Relative review features (NOT proven distortion)", ylabel="Score", xlabel="Original time (s)")
    for ax in axes[:3]:
        ax.axvspan(selected["start"], selected["end"], color="orange", alpha=.18)
        ax.grid(alpha=.1)
    if records:
        d = np.array([[r[key] for key in BANDS] for r in records])
        im = axes[3].imshow(d, aspect="auto", vmin=-12, vmax=12, cmap="RdBu_r")
        axes[3].set(xticks=range(len(BANDS)), yticks=range(len(records)),
                    xticklabels=["20-120", "120-400", "400-2k", "2k-6k", "6k-12k"],
                    yticklabels=[STEM_ENGLISH.get(r["stem"], r["stem"]) for r in records],
                    xlabel="Frequency bands (Hz)", title="Per-stem energy difference vs rest (dB, NOT distortion)")
        for (rr, cc), value in np.ndenumerate(d):
            axes[3].text(cc, rr, f"{value:+.1f}", ha="center", va="center", fontsize=9, color="black" if abs(value)<4 else "white")
        fig.colorbar(im, ax=axes[3], label="Relative dB", shrink=.8)
    fig.suptitle("Sound Analyzer - preliminary review report", fontsize=15)
    b = BytesIO()
    fig.savefig(b, format="png", dpi=135)
    plt.close(fig)
    return b.getvalue()
