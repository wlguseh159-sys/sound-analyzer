"""Sound Analyzer v4 — drum-anchored band balance demonstration.

Run: python -m streamlit run app.py
Important: relative instrument balance is not the same as clipping/distortion.
"""
from __future__ import annotations
from io import BytesIO
import hashlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import soundfile as sf

from analysis_engine import (
    SR, BANDS, info, read_segment, mono, stft, band_normalized_summary,
    clip_events, good_baseline, analyze_balance_chunk, scan_bad_audio,
    consolidate_balance_windows, synthetic_balance_demo, rms_db,
    DETAIL_SR, DETAIL_BANDS, fullband_summary,
)

st.set_page_config(page_title='Sound Analyzer v4 | 드럼 기준 밸런스',page_icon='🥁',layout='wide')

@st.cache_data(show_spinner=False)
def cached_good(good_raw,start,length,good_drum_raw=None):
    sample=read_segment(good_raw,start,length)
    stem=read_segment(good_drum_raw,start,length) if good_drum_raw else None
    return good_baseline(sample,stem),sample


def pcm_wav(data,rate=SR):
    mem=BytesIO()
    sf.write(mem,mono(data),rate,format='WAV',subtype='PCM_16')
    return mem.getvalue()


def band_violin(good,bad):
    table,a,b=fullband_summary(good,bad)
    cells=[]
    for label in a:
        for name,seq in [('Good (기준)',a[label]),('Broken (선택 구간)',b[label])]:
            seq=np.asarray(seq,dtype=float)
            if len(seq)>240:seq=seq[np.linspace(0,len(seq)-1,240,dtype=int)]
            cells.extend({'주파수 대역':label,'STFT 대역 에너지 (dB)':float(v),'음원':name} for v in seq)
    fig=px.violin(pd.DataFrame(cells),x='주파수 대역',y='STFT 대역 에너지 (dB)',
                  color='음원',box=True,points=False,violinmode='group',
                  color_discrete_map={'Good (기준)':'#3986B1','Broken (선택 구간)':'#D87861'})
    fig.update_layout(height=430,legend_title_text='',margin=dict(l=6,r=6,t=24,b=12))
    return fig,pd.DataFrame(table)


def plot_balance(df,segments,baseline,threshold):
    fig=go.Figure()
    if len(df):
        fig.add_trace(go.Scatter(x=df['시작 (초)'],y=df['편차 ΔB (dB)'],name='드럼 대비 편차',
                    mode='lines+markers',marker=dict(size=5),line=dict(color='#3689B2')))
    fig.add_hline(y=threshold,line_dash='dash',line_color='#DA795C',annotation_text=f'+{threshold:g} dB')
    fig.add_hline(y=-threshold,line_dash='dash',line_color='#DA795C',annotation_text=f'-{threshold:g} dB')
    fig.add_hline(y=0,line_dash='dot',line_color='#8A8A8A')
    for seg in segments[:80]:
        fig.add_vrect(x0=seg['start'],x1=seg['end'],fillcolor='#EB7F75',opacity=.16,line_width=0)
    fig.update_layout(height=340,xaxis_title='Broken 원본 시간 (초)',
                      yaxis_title='드럼 대비 나머지 음량 편차 ΔB (dB)',
                      margin=dict(l=8,r=8,t=10,b=20))
    return fig


def plot_stft(audio,absolute_start):
    f,t,z=stft(audio,sr=DETAIL_SR,nfft=2048,hop=512)
    use=(f>=20)&(f<=20000)
    mag=20*np.log10(np.abs(z[use])+1e-8)
    fig=go.Figure(go.Heatmap(x=absolute_start+t,y=f[use],z=mag,
                            zmin=-95,zmax=-20,colorscale='Viridis',
                            colorbar=dict(title='dB')))
    fig.update_layout(height=350,xaxis_title='Broken 원본 시간 (초)',
                      yaxis_title='주파수 (Hz)',margin=dict(l=8,r=8,t=12,b=15))
    return fig


def plot_signal(audio,abstime,segments):
    a=mono(audio);step=max(1,len(a)//5500)
    x=abstime+np.arange(0,len(a),step)/SR
    fig=go.Figure(go.Scatter(x=x,y=a[::step],mode='lines',line=dict(color='#4589A7',width=.65)))
    for seg in segments[:80]:
        fig.add_vrect(x0=seg['start'],x1=seg['end'],fillcolor='#E06F62',opacity=.19,line_width=0)
    fig.update_layout(height=225,xaxis_title='Broken 원본 시간 (초)',
                      yaxis_title='파형 진폭',margin=dict(l=8,r=8,t=10,b=10))
    return fig


def report_png(records,selected,good,bad_piece,baseline,threshold,method):
    """Generate a compact English-labelled, presentation-ready image.
    Linux web hosts may lack a Korean plotting font.
    """
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig,axs=plt.subplots(3,1,figsize=(11,10),gridspec_kw={'height_ratios':[1.2,1.7,1.7]},layout='constrained')
    t=[w['start'] for w in records if w['valid']]
    delta=[w['delta_db'] for w in records if w['valid']]
    axs[0].plot(t,delta,'.-',lw=1,color='#367CB1');axs[0].axhline(threshold,ls='--',color='#D87861')
    axs[0].axhline(-threshold,ls='--',color='#D87861')
    axs[0].axvspan(selected['start'],selected['end'],color='#E9A6A0',alpha=.35)
    axs[0].set(title='A. Drum-anchored relative level change (review, not verified distortion)',
               xlabel='Time in broken recording (s)',ylabel='Delta balance (dB)')
    f,tt,z=stft(bad_piece,sr=DETAIL_SR,nfft=2048,hop=512);keep=(f>20)&(f<20000)
    pcm=axs[1].pcolormesh(tt+selected['start'],f[keep],20*np.log10(abs(z[keep])+1e-8),
                           cmap='viridis',vmin=-95,vmax=-20,shading='auto')
    fig.colorbar(pcm,ax=axs[1],label='STFT magnitude (dB)')
    axs[1].set(title='B. STFT of selected suspect segment',xlabel='Time (s)',ylabel='Frequency (Hz)')
    _,ref,test=fullband_summary(good,bad_piece)
    groups=[];poss=[];clrs=[]
    for i,(label,_,_,_) in enumerate(DETAIL_BANDS):
        groups.extend([ref[label],test[label]]);poss.extend([i*3+1,i*3+2])
        clrs.extend(['#367CB1','#D87861'])
    shapes=axs[2].violinplot(groups,positions=poss,showmedians=True,showextrema=False)
    for body,c in zip(shapes['bodies'],clrs):body.set_facecolor(c);body.set_alpha(.65)
    axs[2].set_xticks([i*3+1.5 for i in range(len(DETAIL_BANDS))],['20-120','120-400','400-2k','2k-6k','6k-10k','10k-20k'])
    axs[2].set(title='C. STFT frequency-band distributions: Good / Broken',
               xlabel='Frequency band (Hz)',ylabel='Band energy (dB)')
    fig.suptitle(f'Sound Analyzer v4  |  {method}',fontsize=14,weight='bold')
    out=BytesIO();FigureCanvasAgg(fig);fig.savefig(out,format='png',dpi=150,bbox_inches='tight')
    plt.close(fig);return out.getvalue()


st.title('🥁 Sound Analyzer v4')
st.caption('공학설계 중간발표 | 드럼(anchor) 기준 상대 밸런스 → ±3 dB 검토 → 선택 구간 STFT → 바이올린 플롯')
st.warning('**밸런스 이상 ≠ 소리 왜곡 확정.** 이 앱은 정상 기준에 비해 드럼 대비 다른 소리의 상대 음량이 달라진 **검토 후보**를 표시합니다. 타격성 추정 모드는 실제 드럼 분리가 아닙니다.')

with st.sidebar:
    st.header('검토 기준')
    threshold=st.slider('드럼 대비 상대 밸런스 편차 (dB)',2.,12.,3.,.5,
                       help='받은 교정 제안서에서는 ±3 dB를 구간별 검토 기준으로 사용했습니다.')
    win=st.selectbox('시간별 분석 단위',[.5,1.,2.],index=1,format_func=lambda x:f'{x:g}초')
    st.divider()
    st.caption('교정 제안서에 있던 ±3 dB는 분석용 의심 기준입니다. 음질 왜곡을 확정하는 보편적 표준은 아닙니다.')

st.subheader('① 정상 Good / 문제 Broken 음원')
demo=st.checkbox('🧪 합성 테스트 음원으로 먼저 시연',value=False,
                 help='4~6초에 드럼은 고정하고 다른 소리만 약 6dB 키운 인공 신호입니다.')
if demo:
    good_x,bad_x,gd_x,bd_x,truth=synthetic_balance_demo(sr=DETAIL_SR)
    good_bytes=pcm_wav(good_x,DETAIL_SR);bad_bytes=pcm_wav(bad_x,DETAIL_SR)
    good_stem_demo=pcm_wav(gd_x,DETAIL_SR);bad_stem_demo=pcm_wav(bd_x,DETAIL_SR)
    good_name='demo_good.wav';bad_name='demo_broken.wav'
else:
    left,right=st.columns(2)
    with left:
        good_file=st.file_uploader('정상 기준 good_take.wav',type=['wav','mp3','flac'],key='g')
    with right:
        bad_file=st.file_uploader('문제 합주 broken_take.mp3',type=['wav','mp3','flac'],key='b')
    good_bytes=good_file.getvalue() if good_file else None
    bad_bytes=bad_file.getvalue() if bad_file else None
    good_name=good_file.name if good_file else 'good.wav'
    bad_name=bad_file.name if bad_file else 'broken.wav'

if not (good_bytes and bad_bytes):
    st.info('Good / Broken 음원을 둘 다 업로드하거나 합성 데모를 체크해 주세요.')
    st.stop()
try:
    gm=info(good_bytes);bm=info(bad_bytes)
except ValueError as e:
    st.error(str(e));st.stop()
st.caption(f'Good {gm["duration"]:.1f}초 · Broken {bm["duration"]:.1f}초 | 기준 길이가 짧으면 전체 Broken의 같은 구간을 보장할 수 없습니다.')

st.subheader('② 드럼 신호를 얻는 방법')
source=st.radio('분석 방식',[
    '빠른 추정: 타격성 신호 HPSS (AI 악기 분리 아님)',
    '정밀 비교: 별도로 분리한 드럼 WAV 사용',
],horizontal=False,index=0)
precise=source.startswith('정밀')
if precise:
    st.info('Good와 Broken을 Demucs 등으로 분리한 뒤 **각각 drums.wav**를 업로드하세요. 두 드럼 파일은 각 원본과 시작 시점·길이가 같아야 합니다. 별도 AI 모델은 이 무료 서버에서 자동 실행하지 않습니다.')
    a,b=st.columns(2)
    with a:
        good_drum_file=st.file_uploader('Good drums.wav',type=['wav','flac'],key='gd')
    with b:
        bad_drum_file=st.file_uploader('Broken drums.wav',type=['wav','flac'],key='bd')
    if not (good_drum_file and bad_drum_file):
        st.warning('분리된 드럼 파일 두 개를 올려야 정밀 비교를 진행할 수 있습니다.')
        st.stop()
    good_stem_raw=good_drum_file.getvalue();bad_stem_raw=bad_drum_file.getvalue()
    if abs(info(good_stem_raw)['duration']-gm['duration'])>1 or abs(info(bad_stem_raw)['duration']-bm['duration'])>1:
        st.error('분리된 드럼 WAV 길이가 원본과 다릅니다. 원본 전체에서 분리한 파일을 사용해 주세요.')
        st.stop()
else:
    st.warning('빠른 추정 모드는 합주에서 **타격성 소리(HPSS)**를 추정하는 방법입니다. 기타의 피킹·보컬 자음도 섞일 수 있어, 이 결과를 실제 드럼 음량이나 악기별 원인이라고 단정하면 안 됩니다.')
    good_stem_raw=good_stem_demo if demo else None
    bad_stem_raw=bad_stem_demo if demo else None
    if demo:
        st.caption('합성 데모만 정답 드럼 트랙이 있으므로 자동으로 실제 드럼 트랙을 사용합니다.')

st.subheader('③ 분석 구간 지정')
mode=st.radio('어디를 분석할까요?',[
    '선택한 짧은 구간 비교', 'Broken 전체 탐색 (Good 기준 통계와 비교)'
],horizontal=True)
full=mode.startswith('Broken 전체')
c1,c2,c3=st.columns(3)
with c1:
    good_start=st.number_input('Good 기준 시작 (초)',0.,max(0.,gm['duration']-.5),0.,.5)
with c2:
    ref_len=st.number_input('Good 기준 길이 (초)',1.,max(1.,gm['duration']-good_start),
                            min(10.,max(1.,gm['duration']-good_start)),.5)
with c3:
    bad_start=st.number_input('Broken 시작 (초)',0.,max(0.,bm['duration']-.5),0.,.5)
if full:
    bad_length=bm['duration']-bad_start
    st.info(f'Broken {bad_start:.1f}~{bm["duration"]:.1f}초를 나누어 탐색합니다. Good {good_start:.1f}~{good_start+ref_len:.1f}초의 **통계적 밸런스 중앙값**만 기준으로 사용하므로, 곡 전개 변화도 이상으로 표시될 수 있습니다.')
else:
    bad_length=st.number_input('Broken 분석 길이 (초)',1.,max(1.,bm['duration']-bad_start),
                               min(10.,max(1.,bm['duration']-bad_start)),.5)
    st.caption('Good는 위의 기준 길이만큼 분석해 상대 음량의 중앙값을 구합니다. 시간대별 1:1 동일 악절 비교가 아닙니다.')

config=(hashlib.sha256(good_bytes).hexdigest()[:16],hashlib.sha256(bad_bytes).hexdigest()[:16],
        float(good_start),float(ref_len),float(bad_start),float(bad_length),threshold,win,source,full)
if st.session_state.get('config_v4')!=config:
    st.session_state['config_v4']=config
    st.session_state.pop('v4_result',None)

if st.button('🔎 드럼 기준 밸런스 분석 실행',type='primary',use_container_width=True):
    try:
        with st.spinner('Good 기준 계산 및 Broken 구간별 밸런스 확인 중...'):
            ref,good_audio=cached_good(good_bytes,good_start,ref_len,good_stem_raw)
            if full:
                records=scan_bad_audio(bad_bytes,bad_start,bad_length,ref['baseline'],threshold,
                                       bad_stem_raw,chunk_sec=10.,win_sec=win)
                bad_preview=read_segment(bad_bytes,bad_start,min(12.,bad_length))
            else:
                bad_preview=read_segment(bad_bytes,bad_start,bad_length)
                bad_stem=read_segment(bad_stem_raw,bad_start,bad_length) if bad_stem_raw else None
                records,_,_,_=analyze_balance_chunk(bad_preview,bad_start,ref['baseline'],threshold,bad_stem,win)
            segments=consolidate_balance_windows(records)
            st.session_state['v4_result']=dict(records=records,segments=segments,
                                               reference=ref,good_audio=good_audio,
                                               bad_preview=bad_preview,abs_bad_start=bad_start)
    except Exception as e:
        st.error(f'분석 중 오류: {e}');st.stop()

if 'v4_result' not in st.session_state:st.stop()
r=st.session_state['v4_result'];records=r['records'];segments=r['segments'];ref=r['reference']
st.subheader('④ 드럼 대비 ±3 dB 이상 의심 구간')
x,y,z=st.columns(3)
x.metric('Good의 드럼 대비 기준',f'{ref["baseline"]:+.1f} dB')
y.metric('의심 구간',f'{len(segments)}개')
z.metric('현재 검토 임계값',f'±{threshold:g} dB')
st.caption(f'분석 방법: {ref["method"]} | 드럼 대비 나머지 = 20log₁₀(RMS(나머지)/RMS(드럼)) | 편차 ΔB = Broken − Good 기준')
frame=pd.DataFrame(records)
plotdf=pd.DataFrame({'시작 (초)':frame['start'], '편차 ΔB (dB)':frame['delta_db']})
st.plotly_chart(plot_balance(plotdf,segments,ref['baseline'],threshold),use_container_width=True)
if not full:
    st.plotly_chart(plot_signal(r['bad_preview'],r['abs_bad_start'],segments),use_container_width=True)
if segments:
    tab=pd.DataFrame([{'번호':i+1,'시작 (초)':v['start'],'끝 (초)':v['end'],
                       '중앙 편차 (dB)':v['avg_delta_db'],'방향':v['direction']} for i,v in enumerate(segments)])
    st.dataframe(tab,use_container_width=True,hide_index=True)
    st.download_button('의심 구간 CSV 저장',tab.to_csv(index=False).encode('utf-8-sig'),
                       'drum_balance_windows.csv','text/csv')
else:
    st.success('선택 조건에서 ±기준을 넘는 구간은 발견되지 않았습니다. 음질이 정상이라고 확정한 결과는 아닙니다.')

st.subheader('⑤ 선택 구간 STFT 및 바이올린 플롯')
if segments:
    ix=st.selectbox('상세 분석할 구간',list(range(len(segments))),
                     format_func=lambda k:f'{k+1}번 · {segments[k]["start"]:.2f}~{segments[k]["end"]:.2f}초 ({segments[k]["avg_delta_db"]:+.1f}dB)')
    selected=segments[ix]
else:
    selected={'start':bad_start,'end':min(bm['duration'],bad_start+min(3.,bad_length)),
              'avg_delta_db':0.,'direction':'검토 기준 이내'}
    st.caption('의심 구간이 없어 첫 구간을 참고용으로 표시합니다.')

start=float(selected['start']);end=min(float(selected['end']),start+12.)
bad_piece=read_segment(bad_bytes,start,max(.4,end-start),sr=DETAIL_SR)
good_detail=read_segment(good_bytes,good_start,ref_len,sr=DETAIL_SR)
st.plotly_chart(plot_stft(bad_piece,start),use_container_width=True)
st.audio(pcm_wav(bad_piece,DETAIL_SR),format='audio/wav')
st.caption('STFT 그래프는 실제 선택 구간의 시간×주파수 성분입니다. 원본 44.1kHz 데이터로 20kHz까지 표시합니다. 같은 대역에 여러 악기가 겹칠 수 있습니다.')
fig,band_df=band_violin(good_detail,bad_piece)
st.plotly_chart(fig,use_container_width=True)
st.dataframe(band_df,use_container_width=True,hide_index=True)

clip=clip_events(bad_piece,sr=DETAIL_SR)
if clip:
    st.warning(f'이 구간에서 디지털 레일 근처의 반복 평탄화 패턴 {len(clip)}건을 찾았습니다. 실제 녹음 클리핑인지는 원본 WAV, 파형 확대와 청취로 재확인이 필요합니다.')
else:
    st.info('이 구간에서 현재 알고리즘의 디지털 클리핑 패턴은 검출되지 않았습니다. 다른 유형의 왜곡이 없다는 뜻은 아닙니다.')

st.subheader('⑥ 발표자료 내보내기')
with st.spinner('발표용 그래프 생성 중...'):
    png=report_png(records,selected,good_detail,bad_piece,ref['baseline'],threshold,
                   'drum stem' if precise or demo else 'HPSS percussive proxy')
st.download_button('📸 결과 PNG 다운로드',png,'Sound_Analyzer_v4_STFT_violin.png','image/png',type='primary')
st.download_button('📋 대역별 수치 CSV 다운로드',band_df.to_csv(index=False).encode('utf-8-sig'),
                   'Sound_Analyzer_v4_bands.csv','text/csv')
with st.expander('결과 판정 기준과 발표 시 설명'):
    st.markdown('''
**1. 밸런스 검토**: 교정 제안서의 예시처럼 `|ΔB| ≥ 3 dB`이면 **드럼 대비 상대적 음량 차이가 있는 구간**으로 표시합니다. 보편적인 왜곡 확정 임계값이 아닙니다.

**2. 음질 왜곡**: 클리핑은 높은 진폭과 평탄한 파형 등 별도의 흔적으로 평가해야 합니다. 녹음 게인 변화, 정상적인 연주 강약, 편곡·다른 악절 등은 밸런스 검토에서 오탐을 만들 수 있습니다.

**3. 악기별 판정**: HPSS는 타격성 추정이고 드럼만 분리하지 않습니다. 업로드한 `drums.wav`는 실제 분리 모델 결과일 수 있으나 누출·인공 잡음이 있을 수 있습니다. 보컬/베이스/기타 중 누가 원인인지 확인하려면 각 악기 stem과 비교 검증이 추가로 필요합니다.

**4. 레벨(dB)**: 디지털 상대 RMS 레벨입니다. 다른 녹음 환경에서 얻은 두 파일의 절대 음량은 그대로 비교하면 안 됩니다.

**5. 발표 추천 표현**: *'드럼을 기준으로 다른 소리와의 상대 밸런스가 3 dB 이상 달라진 검토 구간을 탐지하고, 해당 구간을 STFT·바이올린 플롯으로 시각화했다.'*
''')
