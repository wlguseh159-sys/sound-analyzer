"""STFT-based sound comparison for classroom demonstration.

No claim that musical change proves distortion; results are *review candidates*.
Normal drum vs a mixed band is not a controlled comparison.
"""
from __future__ import annotations
from io import BytesIO
from math import gcd
import numpy as np
import soundfile as sf
from scipy import signal, ndimage

SR = 22050
BANDS = [
    ('20–120 Hz', 20, 120, '킥·베이스 등이 겹침'),
    ('120–400 Hz',120,400,'스네어 몸통·베이스·기타 등이 겹침'),
    ('400–2000 Hz',400,2000,'기타·보컬·스네어 등이 겹침'),
    ('2–6 kHz',2000,6000,'스네어 어택·기타·보컬 등이 겹침'),
    ('6–10 kHz',6000,10000,'심벌·보컬 자음·잡음 등이 겹침'),
]


def info(raw: bytes) -> dict:
    try:
        with sf.SoundFile(BytesIO(raw)) as f:
            return {'sr':f.samplerate,'duration':len(f)/f.samplerate,'channels':f.channels}
    except Exception as e:
        raise ValueError('파일을 읽을 수 없습니다. WAV(PCM) 또는 MP3/FLAC 형식을 확인해 주세요. ' + str(e)) from e


def resample(a, old_sr, new_sr=SR):
    if old_sr == new_sr: return np.asarray(a, dtype=np.float32)
    g = gcd(int(old_sr), int(new_sr))
    return signal.resample_poly(a,new_sr//g,old_sr//g,axis=0).astype(np.float32)


def read_segment(raw:bytes,start=0.,duration=10.,sr=SR):
    try:
        with sf.SoundFile(BytesIO(raw)) as f:
            f.seek(min(len(f), max(0,int(round(float(start)*f.samplerate)))))
            a=f.read(int(round(float(duration)*f.samplerate)),dtype='float32',always_2d=True)
            original_sr=f.samplerate
    except Exception as e:
        raise ValueError('오디오 디코딩에 실패했습니다. MP3 오류가 나면 WAV로 변환해 올려주세요. '+str(e)) from e
    if len(a)<0.4*original_sr: raise ValueError('분석 가능한 데이터가 0.4초 미만입니다. 시작 위치를 바꾸세요.')
    return resample(a,original_sr,sr)


def mono(x):
    x=np.asarray(x,dtype=np.float32)
    return x.mean(axis=1) if x.ndim==2 else x


def stft(x,sr=SR,nfft=1024,hop=256):
    return signal.stft(mono(x),fs=sr,nperseg=nfft,noverlap=nfft-hop,boundary='zeros',padded=True)


def band_frame_db(x,sr=SR,nfft=1024,hop=256):
    f,t,z=stft(x,sr,nfft,hop)
    power=np.abs(z)**2
    d=[]
    for _,low,high,_ in BANDS:
        select=(f>=low)&(f<min(high,sr/2))
        d.append(10*np.log10(power[select].sum(axis=0)+1e-12))
    return t,np.stack(d,axis=1)


def percussive_estimate(x,sr=SR):
    """Median-filter approximation of transients, NOT drum source separation."""
    y=mono(x)
    f,t,z=stft(y,sr)
    mag=np.abs(z)
    harmonic=ndimage.median_filter(mag,size=(1,17),mode='nearest')
    percussive=ndimage.median_filter(mag,size=(17,1),mode='nearest')
    mask=percussive**2/(percussive**2+harmonic**2+1e-18)
    _,estimate=signal.istft(z*mask,fs=sr,nperseg=1024,noverlap=768,boundary=True)
    return np.pad(estimate,(0,max(0,len(y)-len(estimate))))[:len(y)].astype(np.float32)


def onsets(x,sr=SR):
    f,t,z=stft(x,sr)
    mag=np.abs(z)
    flux=np.maximum(np.diff(mag,axis=1,prepend=mag[:,:1]),0).sum(axis=0)
    flux=ndimage.gaussian_filter1d(flux,sigma=1)
    if float(np.max(flux))<1e-8: return np.array([])
    ids,_=signal.find_peaks(flux, height=max(np.quantile(flux,.65),.12*flux.max()),
                             prominence=max(.08*flux.max(),.55*flux.std()),
                             distance=max(1,int(.13*sr/256)))
    return t[ids]


def event_profile(x,t,sr=SR):
    y=mono(x)
    segment=y[max(0,int((t-.015)*sr)):min(len(y),int((t+.14)*sr))]
    if len(segment)<40: return None
    fft=np.fft.rfft(segment*np.hanning(len(segment)),n=4096)
    f=np.fft.rfftfreq(4096,1/sr); p=abs(fft)**2
    features=[]
    for _,lo,hi,_ in BANDS:
        features.append(10*np.log10(p[(f>=lo)&(f<hi)].sum()+1e-10))
    features=np.array(features)
    return features-features.mean() # global gain invariant


def _events_to_windows(events,duration,pre=.2,post=.32,merge_gap=.06):
    events=sorted(events,key=lambda e:e['time']); windows=[]
    for e in events:
        a=max(0.,float(e['time'])-pre);b=min(duration,float(e['time'])+post)
        if windows and a<=windows[-1]['end']+merge_gap:
            windows[-1]['end']=max(b,windows[-1]['end'])
            windows[-1]['events'].append(e)
        else:
            windows.append({'start':a,'end':b,'events':[e]})
    for w in windows:
        w['start']=round(w['start'],3);w['end']=round(w['end'],3)
        w['score']=round(max(float(e['score']) for e in w['events']),2)
        w['method']=', '.join(sorted({e['kind'] for e in w['events']}))
    return windows


def clip_events(x,sr=SR):
    """Detect near-rail repeated samples, not all possible clipping."""
    x=mono(x)
    if len(x)<6 or max(abs(x))<.70: return []
    near=np.abs(x)>.985
    flat=np.abs(np.diff(x))<.00015
    # need 3 consecutive near-rail flat sample differences
    mask=near[1:-2] & near[2:-1] & near[3:] & flat[1:-1] & flat[2:]
    idx=np.flatnonzero(mask)+2
    if not len(idx): return []
    # 60 ms cluster
    bins=np.floor(idx/(sr*.06)).astype(int)
    unique,count=np.unique(bins,return_counts=True)
    return [{'time':float(k*.06),'score':float(12),'kind':'클리핑 패턴 의심'}
            for k,n in zip(unique,count) if n>=2]


def detect_drum_reference(good,bad,sr=SR,sensitivity=1.0,good_is_drum=True):
    """Find outlier drum-like transients in bad full mix versus clean reference.

    With solo drum vs ensemble, other instruments can create false positives.
    """
    gd=mono(good) if good_is_drum else percussive_estimate(good,sr)
    bd=percussive_estimate(bad,sr)
    ghits=onsets(gd,sr);bhits=onsets(bd,sr)
    ref=[event_profile(gd,t,sr) for t in ghits]
    ref=np.array([x for x in ref if x is not None])
    if len(ref)<3: raise ValueError('정상 드럼 기준에서 3회 이상의 타격을 찾지 못했습니다. 드럼 소리가 들리는 파일/구간을 선택하세요.')
    prof=[];validhits=[]
    for t in bhits:
        p=event_profile(bd,t,sr)
        if p is not None:
            validhits.append(t);prof.append(p)
    if not prof: raise ValueError('분석 음원에서 타격성 소리를 찾지 못했습니다.')
    prof=np.asarray(prof); validhits=np.asarray(validhits)
    scores=np.sqrt(np.mean((prof[:,None,:]-ref[None,:,:])**2,axis=2)).min(axis=1)
    rr=np.sqrt(np.mean((ref[:,None,:]-ref[None,:,:])**2,axis=2));np.fill_diagonal(rr,np.inf)
    typical=np.min(rr,axis=1)
    # Adapt to stable accompaniment differences between a solo drum and a full mix.
    # This reduces (but cannot eliminate) the false positives from other instruments.
    ref_limit=max(2.5,float(np.quantile(typical,.8))+1.4)
    mixture_limit=float(np.quantile(scores,.8)) + max(.6,.35*float(np.std(scores)))
    threshold=max(ref_limit,mixture_limit)/max(.4,float(sensitivity))
    events=[{'time':float(t),'score':float(s),'kind':'드럼 특징 편차'}
            for t,s in zip(validhits,scores) if s>threshold]
    events+=clip_events(bad,sr)
    dur=len(bad)/sr
    return {'windows':_events_to_windows(events,dur), 'times':validhits,'scores':scores,
            'threshold':threshold,'reference_n':len(ref),'test_n':len(prof),
            'description':'타격성 신호를 정상 드럼 타격의 주파수 형태와 비교'}


def detect_good_bad(good,bad,sr=SR,sensitivity=1.0):
    """Time-paired good/bad STFT comparison, assumes same passage and tempo.

    Remove global between-take band EQ offset before scoring temporal deviations.
    """
    ta,ga=band_frame_db(good,sr)
    tb,ba=band_frame_db(bad,sr)
    time_len=min(ta[-1],tb[-1],len(good)/sr,len(bad)/sr)
    # group into 0.25sec bins, median is robust to one cymbal frame
    t_starts=np.arange(0,max(time_len-.22,.01),.25)
    features=[]
    for t in t_starts:
        gi=(ta>=t)&(ta<t+.25);bi=(tb>=t)&(tb<t+.25)
        if not gi.any() or not bi.any():continue
        features.append((float(t+.125),np.median(ba[bi],axis=0)-np.median(ga[gi],axis=0)))
    if not features:raise ValueError('비교할 길이가 너무 짧습니다.')
    times=np.array([x[0] for x in features]);diff=np.stack([x[1] for x in features])
    # remove consistent between-take EQ / gain differences: residual anomaly
    typical=np.median(diff,axis=0)
    residual=diff-typical
    scores=np.sqrt((residual**2).mean(axis=1))
    median=float(np.median(scores));mad=float(np.median(np.abs(scores-median)))
    threshold=max(2.0,median+2.8*1.4826*mad)/max(.4,float(sensitivity))
    events=[{'time':float(t),'score':float(s),'kind':'정상 합주 대비 주파수 편차'} for t,s in zip(times,scores) if s>threshold]
    events+=clip_events(bad,sr)
    return {'windows':_events_to_windows(events,time_len,pre=.13,post=.13,merge_gap=.05),
            'times':times,'scores':scores,'threshold':threshold,
            'reference_n':len(t_starts),'test_n':len(t_starts),
            'description':'같은 시점 good/bad 음원의 대역별 STFT 차이를 비교 (시간 정렬 필요)'}


def band_distributions(audio,sr=SR):
    _,db=band_frame_db(audio,sr)
    return {label: db[:,i] for i,(label,_,_,_) in enumerate(BANDS)}


def summarize_bands(reference,candidate,sr=SR):
    a=band_distributions(reference,sr);b=band_distributions(candidate,sr)
    vals=[]
    for label,_,_,instr in BANDS:
        good=float(np.median(a[label]));bad=float(np.median(b[label]))
        vals.append({'주파수 대역':label,'정상 중앙값 (dB)':round(good,2),
                     '의심 중앙값 (dB)':round(bad,2),'차이 (dB)':round(bad-good,2),
                     '악기 겹침 예시':instr})
    return vals,a,b


def synthetic_demo(mode: str = 'band') -> tuple[bytes,bytes]:
    """Deterministic teaching example, synthesized (no copyrighted recording)."""
    from io import BytesIO
    rate=SR;duration=10;n=int(rate*duration)
    times=np.arange(n)/rate
    rng=np.random.default_rng(51)
    drum=np.zeros(n,dtype=np.float32)
    for k,hit_time in enumerate(np.arange(.25,9.9,.5)):
        pos=int(hit_time*rate); m=min(n-pos,int(.20*rate));tt=np.arange(m)/rate
        if k%2==0:
            tone=np.sin(2*np.pi*(75*tt+50*.016*(1-np.exp(-tt/.016))))
            pulse=.47*np.exp(-tt*40)*tone
        else:
            pulse=.24*np.exp(-tt*43)*rng.standard_normal(m)
        pulse+=.1*np.exp(-tt*65)*rng.standard_normal(m)
        drum[pos:pos+m]+=pulse.astype(np.float32)
    bass=.12*np.sin(2*np.pi*(110*times+.18*np.sin(2*np.pi*.15*times)))
    chord=.065*(np.sin(2*np.pi*330*times)+.75*np.sin(2*np.pi*440*times)+.5*np.sin(2*np.pi*660*times))
    good=(drum+bass+chord*(.35+.65*(.5+.5*np.sin(2*np.pi*.24*times)))).astype(np.float32)
    bad=good.copy()
    distorted=(times>=4.0)&(times<5.15)
    bad[distorted]=np.clip(3.3*(bad[distorted]+.16*np.sin(2*np.pi*3900*times[distorted])),-1,1)
    altered=(times>=7.35)&(times<7.75)
    bad[altered]+=.10*np.sin(2*np.pi*7100*times[altered])
    def wav(x):
        buffer=BytesIO()
        sf.write(buffer,x,rate,format='WAV',subtype='PCM_16')
        return buffer.getvalue()
    return wav(drum if mode=='drum' else good),wav(bad)
