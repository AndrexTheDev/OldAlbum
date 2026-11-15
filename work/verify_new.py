#!/usr/bin/env python3
"""Verify the TRACKED deliverables. Never /tmp intermediates."""
import numpy as np, soundfile as sf, subprocess, sys, json
sys.path.insert(0,'work')
import dynamics as dy
SR=44100; FF='work/ffmpeg'
TAGS=('1_I_Need_You','2_Lonesome_Feeling','3_Put_It_On','4_Whats_New_Pussycat')
def lufs(x):
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix='.wav',delete=False) as f: p=f.name
    sf.write(p,x,SR,subtype='FLOAT'); 
    r=subprocess.run([FF,'-hide_banner','-nostats','-i',p,'-af','ebur128=peak=true','-f','null','-'],capture_output=True,text=True)
    os.unlink(p)
    I=TP=None
    for ln in r.stderr.splitlines():
        if 'I:' in ln and 'LUFS' in ln: I=float(ln.split('I:')[1].split('LUFS')[0])
        if 'Peak:' in ln and 'dBFS' in ln: TP=float(ln.split('Peak:')[1].split('dBFS')[0])
    return I,TP
def rel(x,lo,hi):
    m=np.asarray(x,float); m=m.mean(1) if m.ndim>1 else m
    n=8192; k=len(m)//n*n; fr=m[:k].reshape(-1,n)
    W=np.abs(np.fft.rfft(fr*np.hanning(n),axis=1))**2; f=np.fft.rfftfreq(n,1/SR)
    ref=np.median(W[:,(f>300)&(f<3000)].sum(1))
    return float(10*np.log10(np.median(W[:,(f>=lo)&(f<hi)].sum(1))/ref+1e-30))
print('=== 24-bit FLAC Master ===')
print('%-22s %8s %8s %8s %8s %7s %8s'%('song','LUFS','dBTP','12-16k','Crest','Korr','Clip'))
ok=True
for t in TAGS:
    x,_=sf.read(f'masters/{t} (restored master 24bit).flac',dtype='float64')
    I,TP=lufs(x)
    c=int((np.abs(x)>=0.99999).sum())
    corr=np.corrcoef(x[:,0],x[:,1])[0,1]
    pass_l=abs(I+12.0)<=0.1; pass_t=TP<=-1.0+0.05
    if not(pass_l and pass_t and c==0): ok=False
    print('%-22s %8.2f %8.2f %+8.1f %8.1f %8.4f %8d %s'%(t,I,TP,rel(x,12000,16000),
        dy.crest_db(x),corr,c,'OK' if (pass_l and pass_t and c==0) else 'FAIL'))
print()
print('=== MP3 320 ===')
for t in TAGS:
    p=f'restored/{t} (restored).mp3'
    r=subprocess.run([FF,'-hide_banner','-nostats','-i',p,'-af','ebur128=peak=true','-f','null','-'],capture_output=True,text=True)
    I=TP=None
    for ln in r.stderr.splitlines():
        if 'I:' in ln and 'LUFS' in ln: I=float(ln.split('I:')[1].split('LUFS')[0])
        if 'Peak:' in ln and 'dBFS' in ln: TP=float(ln.split('Peak:')[1].split('dBFS')[0])
    good=abs(I+12.0)<=0.15 and TP<=-1.0+0.06
    if not good: ok=False
    print('%-22s %8.2f %8.2f  %s'%(t,I,TP,'OK' if good else 'FAIL'))
print()
print('GESAMT:', 'ALLE PRUEFUNGEN BESTANDEN' if ok else 'FEHLER')
sys.exit(0 if ok else 1)
