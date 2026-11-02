"""Sound Analyzer: upload/record -> review candidates -> optional real AI stems -> frequency map.

Launch: py -m streamlit run app.py
"""
from __future__ import annotations

import hashlib

import numpy as np
import streamlit as st

from audio_engine import (BANDS, STEM_NAMES, analyze_mix, band_table, csv_bytes,
                          inspect_audio, read_segment, wav_bytes, scan_entire_file)
from stem_separator import is_available, separate
from visuals import overview_plot, stem_heatmap, export_report

st.set_page_config(page_title="Sound Analyzer | 공학설계", page_icon="🎛️", layout="wide")
st.title("🎛️ Sound Analyzer")
st.caption("공학설계 · 녹음/업로드 → 전체 STFT → 검토 구간 → AI 악기 분리 → 악기별 주파수 비교")
st.warning("정확도 안내: '특성 변화'는 손상 확정이 아닙니다. 악기 분리에도 누출·인공 잡음이 생길 수 있습니다. 실제 클리핑 판정과 청취 검증을 함께 진행하세요.")

st.subheader("① 음원 입력")
mode = st.radio("입력 방식", ["음원 파일 업로드", "브라우저 마이크 녹음"], horizontal=True)
if mode == "음원 파일 업로드":
    uploaded = st.file_uploader("MP3 / WAV / FLAC 파일 선택", type=["mp3", "wav", "flac"])
else:
    uploaded = st.audio_input("녹음 버튼을 눌러 10초 연주한 뒤 정지하세요", sample_rate=44100)

if uploaded is None:
    st.info("음원을 먼저 올리거나 마이크로 녹음하세요. 처음에는 10초 음원을 추천합니다.")
    st.stop()

raw = uploaded.getvalue()
digest = hashlib.sha256(raw).hexdigest()
format_name = (getattr(uploaded, "name", "") or "").lower()
st.audio(raw, format="audio/mp3" if format_name.endswith(".mp3") else "audio/flac" if format_name.endswith(".flac") else "audio/wav")
try:
    info = inspect_audio(raw)
except Exception as exc:
    st.error(f"음원 파일을 열 수 없습니다: {exc}. MP3에서 오류가 난다면 WAV로 변환해서 다시 시도하세요.")
    st.stop()

m1, m2, m3 = st.columns(3)
m1.metric("전체 길이", f"{info['duration']:.1f}초")
m2.metric("샘플링", f"{info['sample_rate']:,} Hz")
m3.metric("채널", str(info["channels"]))
max_start = max(0.0, info["duration"] - .5)
start_input_key = f"start_{digest[:16]}"
if start_input_key not in st.session_state:
    st.session_state[start_input_key] = 0.0
if info["duration"] > 10.5:
    with st.expander("🔎 전체 음원에서 의심 후보 먼저 찾아보기 (긴 파일용)", expanded=False):
        st.caption("전체 파일을 10초 블록으로 훑고 상대적으로 급격한 음향 변화가 있는 후보를 찾습니다. 확정적인 왜곡 탐지가 아닙니다.")
        if st.button("전체 파일 빠른 탐색 (최대 10분)"):
            try:
                with st.spinner("전체 파일을 순차 분석 중..."):
                    st.session_state["fullscan"] = {"digest": digest, "rows": scan_entire_file(raw)}
            except Exception as exc:
                st.error(f"전체 파일 탐색 실패: {exc}")
        scan_state = st.session_state.get("fullscan")
        if scan_state and scan_state["digest"] == digest and scan_state["rows"]:
            idx = st.selectbox("자동 탐색된 후보", range(len(scan_state["rows"])),
                               format_func=lambda i: f"{scan_state['rows'][i]['start']:.1f}~{scan_state['rows'][i]['end']:.1f}초 · {scan_state['rows'][i]['category']}")
            if st.button("이 후보 주변 10초 구간으로 이동"):
                focus = scan_state["rows"][idx]["start"]
                st.session_state[start_input_key] = float(min(max_start, max(0.0, focus - 3.0)))
                st.rerun()
start = st.number_input("분석 시작 시점 (초)", min_value=0.0, max_value=float(max_start),
                        key=start_input_key, step=0.5)
segment_seconds = min(10.0, info["duration"] - float(start))
st.caption(f"현재 선택: {start:.1f}~{start+segment_seconds:.1f}초 (최대 10초)")
key = (digest, float(start), round(segment_seconds, 2))

st.subheader("② 전체 음원에서 검토 구간 찾기")
if st.button("🔍 STFT 및 이상 의심 구간 분석", type="primary", use_container_width=True):
    try:
        with st.spinner("10초 구간을 분석 중입니다..."):
            audio, sr = read_segment(raw, float(start), segment_seconds)
            result = analyze_mix(audio, sr, offset=float(start))
            st.session_state["current"] = {"key": key, "result": result, "audio": audio, "sr": sr}
            st.session_state.pop("stems", None)
    except Exception as exc:
        st.error(f"음원 분석 실패: {exc}")

state = st.session_state.get("current")
if not state or state["key"] != key:
    st.caption("위 분석 버튼을 누르면 다음 단계가 열립니다.")
    st.stop()

result = state["result"]
if result["clip_count"]:
    st.error(f"높은 진폭에서 파형 평탄화가 의심되는 샘플 {result['clip_count']}개가 관찰됐습니다. 청취 및 원본 확인이 필요합니다.")
else:
    st.info("강한 평탄화(클리핑) 징후는 탐지되지 않았습니다. 아래 후보들은 급격한 음향 변화가 나타난 구간입니다.")

candidates = result["candidates"]
if not candidates:
    st.error("검토 구간을 찾지 못했습니다.")
    st.stop()
options = list(range(len(candidates)))
selected_id = st.selectbox("자세히 볼 구간을 선택하세요", options,
                           format_func=lambda i: f"{candidates[i]['start']:.2f}~{candidates[i]['end']:.2f}초 · {candidates[i]['category']} (상대점수 {candidates[i]['review_score']:.2f})")
selected = candidates[selected_id]
st.plotly_chart(overview_plot(result, selected), use_container_width=True)
st.caption("분홍 음영 = 현재 선택된 구간. 일반적인 드럼 타격, 보컬 진입, 후렴 시작도 특성 변화로 검출될 수 있습니다.")
with st.expander("구간별 검사 수치 보기"):
    st.dataframe(result["rows"], use_container_width=True, hide_index=True)

start_idx = max(0, int(round((selected["start"] - start) * state["sr"])))
end_idx = min(len(state["audio"]), int(round((selected["end"] - start) * state["sr"])))
if end_idx > start_idx:
    st.write("**선택 구간만 다시 듣기**")
    st.audio(wav_bytes(state["audio"][start_idx:end_idx], state["sr"]), format="audio/wav")

st.subheader("③ 악기별 분리 및 주파수 대역 비교")
st.markdown("**기본값은 보컬·드럼·베이스·기타 악기(통합) 4개 그룹**으로 분리합니다. 기타와 피아노를 별도로 나누는 6분리 모델은 실험적이며 정확도가 낮을 수 있습니다.")
model_label = st.radio("분리 그룹", ["4개 그룹 (권장)", "6개 그룹 (실험적)"], horizontal=True)
model = "htdemucs" if model_label.startswith("4") else "htdemucs_6s"
if not is_available():
    st.error("이 컴퓨터에는 AI 악기 분리 패키지 Demucs가 설치되지 않았습니다. 아래 명령어 설치 후 앱을 새로고침하세요.")
    st.code("py -m pip install demucs", language="powershell")
    st.caption("설치 전에도 위 STFT 분석은 가능합니다. 설치 후에는 모델 가중치가 처음 실행할 때 다운로드될 수 있습니다.")
else:
    st.caption("첫 실행에서는 AI 모델 가중치를 다운로드할 수 있습니다. CPU 환경에서는 계산 부담이 큽니다. 모델은 10초 선택 구간만 처리합니다.")
    if st.button("🎚️ 실제 AI 악기 분리 실행", use_container_width=True):
        try:
            with st.spinner("Demucs가 분리 중입니다. 첫 실행 시 모델 파일 다운로드가 필요할 수 있습니다..."):
                stems, sep_sr = separate(state["audio"], state["sr"], model)
                st.session_state["stems"] = {"key": key, "model": model, "stems": stems, "sr": sep_sr}
        except Exception as exc:
            st.error(f"악기 분리에 실패했습니다: {exc}")
            st.caption("ffmpeg 설치, Demucs/torch 의존성, 인터넷 연결(첫 모델 다운로드)을 확인하세요.")

stem_state = st.session_state.get("stems")
records = None
if stem_state and stem_state["key"] == key and stem_state["model"] == model:
    st.success("실제 AI 악기 분리 완료. 선택 구간에서 각 악기의 주파수 변화를 비교합니다.")
    records = band_table(stem_state["stems"], stem_state["sr"], selected["start"], selected["end"], float(start))
    st.plotly_chart(stem_heatmap(records), use_container_width=True)
    st.caption("붉은색(+dB)은 다른 구간에 비해 해당 대역 에너지가 큰 상태입니다. '깨진 악기'를 확정한다는 뜻이 아닙니다.")
    best = sorted(((row["stem"], band, value) for row in records for band, value in row.items() if band in BANDS),
                  key=lambda item: item[2], reverse=True)
    if best:
        name, band, val = best[0]
        st.write(f"**상대적으로 변화가 큰 조합:** {name} / {band} Hz 대역 ({val:+.1f} dB, 다른 구간 대비)")
    st.markdown("**악기별 소리 듣기 및 저장**")
    for name, stem in stem_state["stems"].items():
        with st.expander(f"🎧 {STEM_NAMES.get(name, name)}", expanded=False):
            st.audio(wav_bytes(stem, stem_state["sr"]), format="audio/wav")
            st.download_button(f"{STEM_NAMES.get(name, name)} WAV 내려받기",
                               data=wav_bytes(stem, stem_state["sr"]), file_name=f"{name}.wav",
                               mime="audio/wav", key=f"download_stem_{name}")
else:
    st.info("AI 분리를 실행하면 이 위치에 **악기 × 주파수 대역** 열지도가 나타납니다. 분리하지 않은 음원을 악기별로 분리한 것처럼 표시하지 않습니다.")

st.subheader("④ 결과 다운로드")
png = export_report(result, selected, records)
st.download_button("🖼️ 결과 그래프 PNG 다운로드", data=png, file_name="sound_analysis_report.png", mime="image/png")
st.download_button("📋 구간별 분석 CSV 다운로드", data=csv_bytes(result["rows"]), file_name="review_candidates.csv", mime="text/csv")
if records:
    st.download_button("📊 악기별 대역 CSV 다운로드", data=csv_bytes(records), file_name="stems_frequency_bands.csv", mime="text/csv")
st.caption("실험 결과 보고 시 데이터가 정상 연주인지, 실제 왜곡인지 원본/정상 참조 음원과 비교해서 판단해야 합니다.")
