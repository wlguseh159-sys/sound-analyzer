"""Sound Analyzer: auditable, *heuristic* audio feature extraction.

Important: high spectral energy or large changes do NOT prove distortion.
"""
from __future__ import annotations

from io import BytesIO, StringIO
from math import gcd
import csv

import numpy as np
import soundfile as sf
from scipy import ndimage, signal

# Five display bands, chosen for explanation rather than as instrument boundaries.
BANDS = {
    "초저역 20–120": (20, 120),
    "저중역 120–400": (120, 400),
    "중역 400–2k": (400, 2000),
    "고중역 2k–6k": (2000, 6000),
    "고역 6k–12k": (6000, 12000),
}
STEM_NAMES = {"vocals": "보컬", "drums": "드럼", "bass": "베이스", "other": "기타 악기(묶음)",
              "guitar": "기타(실험)", "piano": "피아노(실험)"}


def inspect_audio(data: bytes) -> dict:
    with sf.SoundFile(BytesIO(data)) as file:
        return {"duration": len(file) / file.samplerate, "sample_rate": file.samplerate,
                "channels": file.channels}


def read_segment(data: bytes, start: float, seconds: float = 10.0) -> tuple[np.ndarray, int]:
    if start < 0 or seconds <= 0:
        raise ValueError("분석 구간을 확인해 주세요.")
    with sf.SoundFile(BytesIO(data)) as file:
        file.seek(min(int(round(start * file.samplerate)), len(file)))
        audio = file.read(int(round(seconds * file.samplerate)), dtype="float32", always_2d=True)
        sr = file.samplerate
    if len(audio) < int(sr * 0.5):
        raise ValueError("최소 0.5초 이상을 선택해야 합니다.")
    return audio, sr


def mono(stereo: np.ndarray) -> np.ndarray:
    return np.mean(stereo, axis=1) if stereo.ndim == 2 else stereo


def resample_audio(data: np.ndarray, old_sr: int, new_sr: int) -> np.ndarray:
    if old_sr == new_sr:
        return data
    d = gcd(old_sr, new_sr)
    return signal.resample_poly(data, new_sr // d, old_sr // d, axis=0).astype(np.float32)


def spectrum(audio: np.ndarray, sr: int) -> dict:
    wave = mono(audio).astype(np.float64)
    n_fft = 2048 if sr >= 16000 else 1024
    hop = n_fft // 4
    frequencies, times, stft = signal.stft(
        wave, fs=sr, nperseg=n_fft, noverlap=n_fft-hop,
        boundary="zeros", padded=True, window="hann",
    )
    power = np.abs(stft) ** 2
    decibels = 20 * np.log10(np.maximum(np.abs(stft), 1e-10) / max(np.abs(stft).max(), 1e-10))
    return {"wave": wave, "sr": sr, "freqs": frequencies, "times": times,
            "power": power, "db": decibels, "hop": hop, "n_fft": n_fft}


def _robust_outlier(feature: np.ndarray) -> np.ndarray:
    # Zero for ordinary values; positive for unusually high values.
    med = float(np.median(feature))
    mad = float(np.median(np.abs(feature - med)))
    scale = max(mad * 1.4826, float(np.std(feature)) * 0.25, 1e-9)
    return np.clip((feature - med) / scale - 1.0, 0, 6) / 6


def _plateau_events(wave: np.ndarray) -> np.ndarray:
    """Samples at nearly same (large) amplitude 3+ consecutive times.

    Flat topped waveform heuristic, not a definitive analog-clipping test.
    """
    if len(wave) < 4 or np.max(np.abs(wave)) < .5:
        return np.zeros(len(wave), dtype=bool)
    peak = float(np.max(np.abs(wave)))
    high = np.abs(wave) >= max(0.6, peak * .985)
    close = np.abs(np.diff(wave)) < 2e-5
    events = np.zeros(len(wave), dtype=bool)
    idx = np.where(close[:-1] & close[1:] & high[:-2] & high[1:-1] & high[2:])[0]
    for i in idx:
        events[i:i+3] = True
    return events


def analyze_mix(audio: np.ndarray, sr: int, offset: float = 0.0) -> dict:
    spec = spectrum(audio, sr)
    wave = spec["wave"]
    p = spec["power"]
    f, t = spec["freqs"], spec["times"]
    total = np.maximum(np.sum(p, axis=0), 1e-15)
    hf = np.sum(p[f >= 4000], axis=0) / total
    # Normalized positive spectral flux: common on drum hits, also can show artifacts.
    pdiff = np.diff(p, axis=1, prepend=p[:, :1])
    spectral_flux = np.maximum(pdiff, 0).sum(axis=0) / total
    spectral_flatness = np.exp(np.mean(np.log(p + 1e-15), axis=0)) / np.maximum(np.mean(p, axis=0), 1e-15)
    rms_samples = np.sqrt(np.maximum(ndimage.uniform_filter1d(wave**2, size=spec["n_fft"]), 0))
    centers = np.minimum(np.round(t * sr).astype(int), len(wave)-1)
    rms = np.maximum(rms_samples[centers], 1e-8)
    clip_events = _plateau_events(wave)
    # A hard clipping candidate requires plateau-like shape near peak.
    sample_clip_times = np.flatnonzero(clip_events) / sr
    # Spectral features only nominate 'review' candidates, not damaged regions.
    score = 0.48 * _robust_outlier(spectral_flux) + 0.32 * _robust_outlier(hf) + 0.20 * _robust_outlier(spectral_flatness)
    # Use 0.5 s bins; avoid overlapping bin endpoints.
    rows = []
    duration = len(wave) / sr
    for begin in np.arange(0, duration, 0.5):
        finish = min(begin + 0.5, duration)
        sel = (t >= begin) & (t < finish)
        if not sel.any():
            continue
        clip_count = int(np.sum((sample_clip_times >= begin) & (sample_clip_times < finish)))
        rows.append({"start": round(float(offset + begin), 2),
                     "end": round(float(offset + finish), 2),
                     "review_score": round(float(np.percentile(score[sel], 80)), 4),
                     "rms_dbfs_approx": round(float(np.mean(20*np.log10(rms[sel]))), 2),
                     "clip_samples": clip_count,
                     "category": "평탄화/클리핑 의심" if clip_count >= 3 else "주파수 특성 변화"})
    rows.sort(key=lambda r: (r["clip_samples"] >= 3, r["review_score"]), reverse=True)
    candidates = rows[:min(5, len(rows))]
    return {"spec": spec, "score": score, "rms": rms, "duration": duration,
            "max_amplitude": float(np.abs(wave).max()),
            "near_rail_percent": float(np.mean(np.abs(wave) >= .995) * 100),
            "clip_count": int(clip_events.sum()), "rows": rows,
            "candidates": candidates, "offset": offset}


def band_comparison(spec: dict, start_abs: float, end_abs: float, offset: float = 0.0) -> dict:
    """Per-band dB relative to rest of same segment, not 'distortion amount'."""
    times, freqs, power = spec["times"], spec["freqs"], spec["power"]
    active = (times >= start_abs - offset) & (times < end_abs - offset)
    if not active.any():
        active[np.argmin(np.abs(times - (start_abs + end_abs)/2 + offset))] = True
    other = ~active
    if not other.any():
        other = active
    comparison = {}
    for label, (low, high) in BANDS.items():
        binmask = (freqs >= low) & (freqs < min(high, spec["sr"]/2))
        if not binmask.any():
            comparison[label] = float("nan")
            continue
        selected_energy = float(power[binmask][:, active].sum(axis=0).mean())
        reference_energy = float(power[binmask][:, other].sum(axis=0).mean())
        comparison[label] = float(np.clip(10*np.log10((selected_energy+1e-12)/(reference_energy+1e-12)), -30, 30))
    return comparison


def band_table(stems: dict[str, np.ndarray], sr: int, begin: float, end: float, offset: float) -> list[dict]:
    records = []
    for name, wave in stems.items():
        spec = spectrum(wave, sr)
        comparison = band_comparison(spec, begin, end, offset)
        records.append({"stem": STEM_NAMES.get(name, name), **comparison})
    return records


def csv_bytes(rows: list[dict]) -> bytes:
    if not rows:
        return b""
    s = StringIO()
    writer = csv.DictWriter(s, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return s.getvalue().encode("utf-8-sig")


def wav_bytes(audio: np.ndarray, sr: int) -> bytes:
    f = BytesIO()
    sf.write(f, audio, sr, format="WAV", subtype="PCM_16")
    return f.getvalue()


def scan_entire_file(data: bytes, *, block_seconds: float = 10.0, max_seconds: float = 600.0) -> list[dict]:
    """Stream through a file looking for relative feature-change candidates.

    Each 10 s chunk is normalized independently; scores across chunks are ONLY
    exploratory rankings, not calibrated distortion probabilities.
    """
    all_rows = []
    with sf.SoundFile(BytesIO(data)) as file:
        sr = file.samplerate
        total = min(float(len(file) / sr), max_seconds)
        for offset in np.arange(0.0, total, block_seconds):
            count = int(min(block_seconds, total - offset) * sr)
            block = file.read(count, dtype="float32", always_2d=True)
            if len(block) < int(.5 * sr):
                continue
            result = analyze_mix(block, sr, float(offset))
            all_rows.extend([r for r in result["candidates"] if r["rms_dbfs_approx"] > -38 or r["clip_samples"] >= 3][:2])
    all_rows.sort(key=lambda item: (item["clip_samples"] >= 3, item["review_score"]), reverse=True)
    return all_rows[:10]
