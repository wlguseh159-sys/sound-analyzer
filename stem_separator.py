"""Run genuine AI source separation through optional Demucs package.

No frequency-split masquerades as instrument stems.
"""
from __future__ import annotations

from pathlib import Path
import importlib.util
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf
from audio_engine import resample_audio


def is_available() -> bool:
    return importlib.util.find_spec("demucs") is not None


def separate(audio: np.ndarray, sr: int, model: str = "htdemucs") -> tuple[dict[str, np.ndarray], int]:
    if not is_available():
        raise RuntimeError("Demucs가 설치되지 않았습니다. 터미널에서: py -m pip install demucs")
    if model not in ("htdemucs", "htdemucs_6s"):
        raise ValueError("지원하지 않는 분리 모델입니다.")
    stereo = audio if audio.ndim == 2 else audio[:, None]
    stereo = stereo.astype(np.float32)
    if stereo.shape[1] == 1:
        stereo = np.repeat(stereo, 2, axis=1)
    elif stereo.shape[1] > 2:
        stereo = stereo[:, :2]
    target_sr = 44100
    stereo = resample_audio(stereo, sr, target_sr)
    # tmpdir is private to this analysis. Demucs may download weights on first use.
    with tempfile.TemporaryDirectory(prefix="audio_analysis_") as folder:
        folder = Path(folder)
        input_path = folder / "input.wav"
        output_root = folder / "separated"
        sf.write(str(input_path), stereo, target_sr, subtype="PCM_16")
        command = [sys.executable, "-m", "demucs", "-n", model, "-d", "cpu",
                   "-o", str(output_root), str(input_path)]
        try:
            execution = subprocess.run(command, capture_output=True, text=True, timeout=1200)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("악기 분리 실행 시간이 초과됐습니다. 짧은 음원으로 다시 시도하세요.") from exc
        if execution.returncode != 0:
            message = (execution.stderr or execution.stdout or "알 수 없는 오류")[-1700:]
            raise RuntimeError("Demucs 실행 실패: " + message)
        stem_folder = output_root / model / "input"
        names = ["vocals", "drums", "bass", "other"]
        if model == "htdemucs_6s":
            names.extend(["guitar", "piano"])
        result: dict[str, np.ndarray] = {}
        for key in names:
            path = stem_folder / (key + ".wav")
            if not path.exists():
                raise RuntimeError(f"악기별 출력 파일이 없습니다: {path.name}")
            stem, stem_sr = sf.read(path, dtype="float32", always_2d=True)
            if stem_sr != target_sr:
                raise RuntimeError("분리된 파일의 샘플링 주파수가 예상과 다릅니다.")
            # Length may differ by a few samples because of resampling.
            length = len(stereo)
            if len(stem) > length:
                stem = stem[:length]
            elif len(stem) < length:
                stem = np.pad(stem, ((0, length - len(stem)), (0, 0)))
            result[key] = stem
    return result, target_sr
