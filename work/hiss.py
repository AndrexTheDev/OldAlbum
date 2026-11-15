"""Rauschhoerbarkeit: Oberband-Pegel in den LEISESTEN Frames."""
import numpy as np, soundfile as sf

SR=44100
def mono(y):
    y=np.asarray(y,dtype=np.float64)
    return y.mean(1) if y.ndim>1 else y

def hiss(mono_x, n=2048):
    """HF-Pegel relativ zum Mittelband, gemessen NUR in den leisesten 10% Frames.
    Das ist der Bereich, in dem Rauschen hoerbar wird."""
    k=len(mono_x)//n*n; fr=mono_x[:k].reshape(-1,n)
    W=np.abs(np.fft.rfft(fr*np.hanning(n),axis=1))**2
    f=np.fft.rfftfreq(n,1/SR)
    hf=W[:,(f>=8000)&(f<14000)].sum(1)
    mid=W[:,(f>=300)&(f<3000)].sum(1)
    tot=W.sum(1)
    q=np.argsort(tot)[:max(1,int(0.10*len(tot)))]     # leiseste 10%
    rel=10*np.log10((hf[q].mean()+1e-30)/(mid[q].mean()+1e-30))
    return float(rel)

def flat(m,lo=8000,hi=14000,n=2048):
    k=len(m)//n*n; fr=m[:k].reshape(-1,n)
    W=np.abs(np.fft.rfft(fr*np.hanning(n),axis=1))**2
    f=np.fft.rfftfreq(n,1/SR); b=W[:,(f>=lo)&(f<hi)]
    gm=np.exp(np.mean(np.log(b+1e-30),1)); am=b.mean(1)
    return float(10*np.log10(np.median(gm/(am+1e-30))+1e-30))
