"""Sound Analyzer v3: normal-reference comparison, suspicious-window STFT, violin plots.

Created for an engineering design midterm demo, not a calibrated distortion meter.
"""
from __future__ import annotations
from io import BytesIO
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import streamlit as st

from analysis_engine import (
    SR,BANDS,info,read_segment,stft,mono,detect_drum_reference,
    detect_good_bad,summarize_bands,synthetic_demo,
)

HERE=Path(__file__).resolve().parent
st.set_page_config(page_title='Sound Analyzer v3 · 공학설계',page_icon='🥁',layout='wide')
st.markdown('''<style>
.block-container{padding-top:2rem;max-width:1200px}
h1{letter-spacing:-.03em}
[data-testid="stMetric"]{background:rgba(128,128,128,.07);padding:1rem;border-radius:.7rem}
</style>''', unsafe_allow_html=True)


def as_wav(x,sr=SR):
    import soundfile as sf
    buf=BytesIO()
    sf.write(buf,np.asarray(x,dtype=np.float32),sr,format='WAV',subtype='PCM_16')
    return buf.getvalue()


def spectrum_fig(x,origin,sr=SR):
    f,t,z=stft(x,sr=sr,nfft=2048,hop=256)
    idx=(f>=20)&(f<=10000)
    power=20*np.log10(np.abs(z[idx])+1e-7)
    fig=go.Figure(go.Heatmap(x=t+origin,y=f[idx],z=power,zmin=-95,zmax=-20,
                              colorscale='Viridis',colorbar=dict(title='dB')))
    fig.update_layout(height=360,margin=dict(l=4,r=4,t=16,b=12),
                      xaxis_title='시간 (초)',yaxis_title='주파수 (Hz)')
    return fig


def wave_fig(audio,windows,offset,sr=SR):
    y=mono(audio)
    step=max(1,len(y)//4500)
    t=offset+np.arange(0,len(y),step)/sr
    fig=go.Figure(go.Scatter(x=t,y=y[::step],mode='lines',name='음원 파형',
                             line=dict(color='#508bbd',width=1)))
    for w in windows:
        fig.add_vrect(x0=offset+w['start'],x1=offset+w['end'],
                      fillcolor='red',opacity=.17,line_width=0)
    fig.update_layout(height=245,xaxis_title='원본 음원 시간 (초)',yaxis_title='진폭',
                      margin=dict(l=5,r=5,t=10,b=10))
    return fig


def score_fig(result,offset):
    times=result['times'];scores=result['scores'];thr=result['threshold']
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=offset+times,y=scores,name='비교 점수',
                             mode='lines+markers',marker=dict(size=5)))
    fig.add_hline(y=thr,line_dash='dash',line_color='#d97757',annotation_text='검토 기준선')
    fig.update_layout(height=230,margin=dict(l=5,r=10,t=10,b=10),
                      xaxis_title='분석 음원 시간 (초)',yaxis_title='차이 점수 (상대값)')
    return fig


def violin_fig(reference,candidate):
    rows=[]
    for label in reference:
        for title,vals in [('정상 기준',reference[label]),('이상 의심',candidate[label])]:
            values=np.asarray(vals,dtype=float)
            # enough data for distribution without sending thousands of data points to browser
            idx=np.linspace(0,len(values)-1,min(len(values),130)).astype(int)
            rows.extend([{'주파수 대역':label,'STFT 대역 에너지 (dB)':float(v),'구분':title} for v in values[idx]])
    df=pd.DataFrame(rows)
    fig=px.violin(df,x='주파수 대역',y='STFT 대역 에너지 (dB)',color='구분',
                  box=True,points=False,violinmode='group',
                  color_discrete_map={'정상 기준':'#3C85B5','이상 의심':'#E07861'})
    fig.update_layout(height=420,margin=dict(l=10,r=5,t=25,b=10),legend_title_text='')
    return fig


def png_report(audio,offset,w,good,reference_name,refdb,baddb):
    # All labels in PNG are in English to avoid unsupported Korean fonts on Linux.
    f,t,z=stft(audio,nfft=2048,hop=256)
    keep=(f>=20)&(f<=10000)
    fig,axes=plt.subplots(3,1,figsize=(11,11),layout='constrained',
                          gridspec_kw={'height_ratios':[1,2.3,1.7]})
    y=mono(audio)
    step=max(1,len(y)//4000)
    axes[0].plot(offset+np.arange(0,len(y),step)/SR,y[::step],lw=.6)
    axes[0].axvspan(offset+w['start'],offset+w['end'],alpha=.23,color='orangered')
    axes[0].set(title='A. Waveform and selected review window',xlabel='Time (s)',ylabel='Amplitude')
    f2,t2,z2=stft(audio[int(w['start']*SR):int(w['end']*SR)],nfft=2048,hop=256)
    sel=(f2>=20)&(f2<=10000)
    power=20*np.log10(abs(z2[sel])+1e-7)
    heat=axes[1].pcolormesh(t2+offset+w['start'],f2[sel],power,shading='auto',
                             cmap='viridis',vmin=-95,vmax=-20)
    fig.colorbar(heat,ax=axes[1],label='Magnitude (dB)')
    axes[1].set(title='B. STFT of selected window',xlabel='Time (s)',ylabel='Frequency (Hz)')
    group=[];positions=[];colours=[]
    for i,(name,_,_,_) in enumerate(BANDS):
        group.extend([refdb[name],baddb[name]])
        positions.extend([3*i+1,3*i+2])
        colours+=['#4E86AA','#D8795B']
    violin=axes[2].violinplot(group,positions=positions,showmedians=True,showextrema=False)
    for obj,c in zip(violin['bodies'],colours):obj.set_facecolor(c);obj.set_alpha(.65)
    axes[2].set_xticks([3*i+1.5 for i in range(len(BANDS))],
                       ['20-120','120-400','400-2k','2k-6k','6k-10k'])
    axes[2].set(title=f'C. STFT band-energy distributions: {reference_name} vs candidate',
                xlabel='Frequency band (Hz)',ylabel='Band energy (relative dB)')
    axes[2].plot([],[],color='#4E86AA',lw=8,label='Good / baseline')
    axes[2].plot([],[],color='#D8795B',lw=8,label='Candidate')
    axes[2].legend(fontsize=8,loc='lower right')
    fig.suptitle('Sound Analyzer v3 | Suspected segment is NOT verified clipping',weight='bold')
    output=BytesIO();fig.savefig(output,format='png',dpi=155,bbox_inches='tight')
    plt.close(fig)
    return output.getvalue()


st.title('🥁 Sound Analyzer v3')
st.caption('공학설계 중간발표 · 정상 기준 → 의심 구간 탐지 → 해당 구간 STFT → 주파수 대역 바이올린 플롯')
st.warning('⚠️ 검출된 것은 **기준과 다른 이상 의심 구간**입니다. 주파수 대역만으로 특정 악기 고장이나 실제 음질 왜곡을 확정할 수 없습니다.')

with st.sidebar:
    st.header('분석 설정')
    mode=st.radio('비교 방식',[
        '정상 합주 vs 문제 합주 (권장)',
        '정상 드럼 vs 문제 합주 (실험)',
    ],help='정상 드럼 단독 vs 전체 합주는 다른 악기 소리 때문에 오탐할 수 있습니다.')
    sensitivity=st.slider('이상 탐지 민감도',.6,2.0,1.0,step=.1,
                           help='높이면 더 많은 후보를 표시하지만 오탐도 늘 수 있습니다.')
    st.divider()
    st.markdown('**실험 설계 팁**')
    st.caption('good/bad는 같은 곡의 같은 구간, 유사한 볼륨·마이크 위치여야 합니다. 드럼 기준 비교는 드럼 단독 트랙을 쓰는 것이 좋습니다.')

st.subheader('① 정상 기준 및 문제 음원 입력')
use_demo=st.checkbox('🎧 제공된 합성 good/bad 테스트 음원으로 시연하기',value=False,
                     help='실제 합주 음원과 구분되는 교육용 합성 파일입니다.')
left,right=st.columns(2)
if use_demo:
    drum_demo=mode.startswith('정상 드럼')
    good_bytes,bad_bytes=synthetic_demo('drum' if drum_demo else 'band')
    left.success('정상 기준: 합성 드럼' if drum_demo else '정상 기준: 합성 good 합주')
    right.success('문제 음원: 합성 bad 합주')
else:
    with left:
        st.markdown('**정상 기준 (good)**')
        st.caption('정상 합주 또는 기준 드럼 녹음')
        good_upload=st.file_uploader('Good 음원',type=['wav','mp3','flac'],key='good')
        good_bytes=good_upload.getvalue() if good_upload is not None else None
    with right:
        st.markdown('**문제 음원 (bad)**')
        st.caption('실제 합주 녹음')
        choice=st.radio('문제 음원 입력 방식',['파일 업로드','브라우저 녹음'],horizontal=True)
        if choice=='파일 업로드':
            bad_upload=st.file_uploader('Bad 음원',type=['wav','mp3','flac'],key='bad')
        else:
            bad_upload=st.audio_input('Bad 합주 녹음',sample_rate=44100)
        bad_bytes=bad_upload.getvalue() if bad_upload is not None else None

if not (good_bytes and bad_bytes):
    st.info('정상 기준 파일과 문제 음원을 각각 올리거나, 위의 **합성 테스트 음원으로 시연하기**를 체크한 후 분석을 시작하세요.')
    st.stop()

try:
    good_meta=info(good_bytes);bad_meta=info(bad_bytes)
except ValueError as exc:
    st.error(str(exc));st.stop()

with st.expander('업로드한 음원 확인',expanded=False):
    c1,c2=st.columns(2)
    with c1:
        st.write(f'Good · {good_meta["duration"]:.1f}초 · {good_meta["sr"]} Hz')
        good_format='audio/wav' if use_demo else ({'wav':'audio/wav','mp3':'audio/mpeg','flac':'audio/flac'}).get(good_upload.name.rsplit('.',1)[-1].lower(),'audio/wav')
        st.audio(good_bytes,format=good_format)
    with c2:
        st.write(f'Bad · {bad_meta["duration"]:.1f}초 · {bad_meta["sr"]} Hz')
        bad_format='audio/wav' if use_demo or choice=='브라우저 녹음' else ({'wav':'audio/wav','mp3':'audio/mpeg','flac':'audio/flac'}).get(bad_upload.name.rsplit('.',1)[-1].lower(),'audio/wav')
        st.audio(bad_bytes,format=bad_format)

col1,col2,col3=st.columns(3)
max_bad=max(0.,bad_meta['duration']-.6)
max_good=max(0.,good_meta['duration']-.6)
with col1:
    start_bad=st.number_input('Bad 음원 분석 시작 (초)',min_value=0.,max_value=float(max_bad),value=0.,step=.5)
with col2:
    start_good=st.number_input('Good 음원 시작 (초)',min_value=0.,max_value=float(max_good),value=0.,step=.5)
with col3:
    max_sec=max(.4,min(20.,bad_meta['duration']-start_bad,good_meta['duration']-start_good))
    length=st.number_input('분석 길이 (초)',min_value=.4,max_value=float(max_sec),value=float(min(10.,max_sec)),step=.5)

# Changing audio, mode, or the selection must invalidate the previous chart.
config_key=(hashlib.sha256(good_bytes).hexdigest(),hashlib.sha256(bad_bytes).hexdigest(),
            mode,round(start_bad,3),round(start_good,3),round(length,3),round(sensitivity,2))
if st.session_state.get('v3_config_key')!=config_key:
    st.session_state.pop('result_v3',None)
    st.session_state['v3_config_key']=config_key

if mode.startswith('정상 드럼'):
    st.info('드럼 기준 실험: **Good 드럼 타격 형태**를 기준으로 Bad 합주의 타격성 소리를 비교합니다. 기타·보컬 등의 영향이 포함될 수 있습니다.')
else:
    st.info('정상 합주 비교: Good/Bad가 **같은 연주 위치·박자·악기 구성**일 때 비교가 가장 유효합니다. 시작 시간을 맞추세요.')

if st.button('🔎 정상 기준과 비교해 의심 구간 찾기',type='primary',use_container_width=True):
    try:
        with st.spinner('음원을 읽고 주파수 특징을 비교하는 중...'):
            good=read_segment(good_bytes,start_good,length)
            bad=read_segment(bad_bytes,start_bad,length)
            if mode.startswith('정상 드럼'):
                detection=detect_drum_reference(good,bad,sensitivity=sensitivity,good_is_drum=True)
            else:
                detection=detect_good_bad(good,bad,sensitivity=sensitivity)
            st.session_state['result_v3']={
                'good':good,'bad':bad,'det':detection,'start_bad':start_bad,
                'start_good':start_good,'drum_mode':mode.startswith('정상 드럼'),
                'reference_name':'normal drums' if mode.startswith('정상 드럼') else 'aligned good band',
            }
    except Exception as exc:
        st.error(f'분석 실패: {exc}')
        st.stop()

if 'result_v3' not in st.session_state:
    st.stop()

r=st.session_state['result_v3']
det=r['det'];windows=det['windows'];good=r['good'];bad=r['bad'];offset=r['start_bad']
st.subheader('② 기준과 다른 구간 자동 탐지')
x,y,z=st.columns(3)
x.metric('의심 구간 수',len(windows))
y.metric('검토 기준 점수',f'{det["threshold"]:.2f}')
z.metric('검출 방법','드럼 타격 비교' if r['drum_mode'] else 'good/bad STFT 비교')
st.plotly_chart(wave_fig(bad,windows,offset),use_container_width=True)
st.plotly_chart(score_fig(det,offset),use_container_width=True)

if not windows:
    st.success('선택한 조건에서 기준을 초과한 의심 구간을 발견하지 못했습니다. **정상이라는 확정 판정은 아닙니다.** 민감도를 조정하거나 다른 구간을 검사할 수 있습니다.')
    st.stop()

wdf=pd.DataFrame([{'번호':i+1,'시작 (초)':round(offset+w['start'],2),
                  '끝 (초)':round(offset+w['end'],2),'차이 점수':w['score'],
                  '이유':w['method']} for i,w in enumerate(windows)])
st.dataframe(wdf,use_container_width=True,hide_index=True)
st.download_button('의심 구간 CSV 저장',data=wdf.to_csv(index=False).encode('utf-8-sig'),
                   file_name='suspected_windows.csv',mime='text/csv')
choice=st.selectbox('자세히 분석할 구간 선택',options=list(range(len(windows))),
                     format_func=lambda i:f'{i+1}번 · {offset+windows[i]["start"]:.2f} ~ {offset+windows[i]["end"]:.2f}초')
w=windows[choice]
start_idx=int(w['start']*SR);end_idx=int(w['end']*SR)
# Add 0.05 second edges for spectrogram while preserving selected region.
start_idx=max(0,start_idx);end_idx=min(len(bad),max(end_idx,start_idx+int(.15*SR)))
bad_piece=bad[start_idx:end_idx]
if r['drum_mode']:
    good_piece=good  # full clean reference used for violin distribution
    comparison_title='정상 드럼 전체 기준 / Bad 선택 구간 (녹음 구성 다름)'
else:
    good_piece=good[min(start_idx,len(good)-1):min(end_idx,len(good))]
    comparison_title='같은 시간 good 합주 / bad 합주'

st.subheader('③ 선택한 구간 STFT')
st.caption(f'Bad 원본 기준: {offset+w["start"]:.2f} ~ {offset+w["end"]:.2f}초 · 이 구간만 상세 STFT 표시')
st.plotly_chart(spectrum_fig(bad_piece,offset+w['start']),use_container_width=True)
st.audio(as_wav(bad_piece),format='audio/wav')

st.subheader('④ STFT 기반 바이올린 플롯 · 주파수 대역 차이')
st.caption(comparison_title)
summary,refdist,baddist=summarize_bands(good_piece,bad_piece)
st.plotly_chart(violin_fig(refdist,baddist),use_container_width=True)
st.dataframe(pd.DataFrame(summary),use_container_width=True,hide_index=True)
top=max(summary,key=lambda x:abs(x['차이 (dB)']))
st.markdown(f'**선택 구간에서 차이가 가장 큰 대역:** {top["주파수 대역"]} (에너지 중앙값 차이 {top["차이 (dB)"]:+.1f} dB)')
st.caption('이는 STFT 대역별 에너지의 상대 비교입니다. 기타·보컬·드럼은 동일 대역에 겹치므로 실제로 어떤 악기가 깨졌는지는 확정할 수 없습니다.')
if r['drum_mode']:
    st.warning('기준은 **드럼 단독**, 비교 대상은 **합주 전체**입니다. 바이올린 플롯 차이에 드럼 이외 소리도 포함됩니다. 발표에서는 실험적 지표로 소개하세요.')

st.subheader('⑤ 발표용 결과 다운로드')
with st.spinner('발표에 사용할 PNG 결과 그림을 생성하고 있습니다...'):
    png=png_report(bad,offset,w,good,r['reference_name'],refdist,baddist)
st.download_button('📸 발표용 STFT + 바이올린 플롯 PNG 저장',data=png,
                   file_name='sound_analyzer_v3_report.png',mime='image/png',type='primary')
st.download_button('📋 대역별 수치 CSV 저장',data=pd.DataFrame(summary).to_csv(index=False).encode('utf-8-sig'),
                   file_name='sound_analyzer_v3_band_comparison.csv',mime='text/csv')
with st.expander('결과 해석과 한계'):
    st.markdown('''
- **검토 구간**: 정상 기준과 대역 에너지 형태나 드럼 타격 특징이 달라진 시점입니다. 깨짐 확정이 아닙니다.
- **STFT**: 선택 구간에서 시간과 주파수의 변화를 보여줍니다.
- **바이올린 플롯**: STFT 각 프레임에서 구한 저역~고역 에너지 분포를 정상/의심 상태별로 비교합니다.
- **드럼 기준**: 타격성 신호 분리는 드럼만 정확히 분리하는 기술이 아닙니다. 신뢰도를 높이려면 드럼 개별 녹음 또는 전문 소스 분리와 대조해야 합니다.
- **정밀 검증**: 실제 왜곡 확정에는 무왜곡 원본과의 동기화, 클리핑 징후, 청취 및 측정 환경 통제가 필요합니다.
''')
