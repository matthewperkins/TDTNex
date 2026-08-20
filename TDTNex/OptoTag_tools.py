from numba import njit
import numpy as np
from math import floor,sqrt

@njit
def descritized_spike_train(spike_times, bins):
    has_spk = np.digitize(spike_times,bins)
    dst = np.zeros(bins.shape,dtype=np.bool)
    dst[has_spk] = True
    return dst
    
@njit
def descritized_spike_raster(event_times,dst,dt,Wn):
    '''event_times, times of each trigger (seconds)
    dst, discretized spike raster, with each bin dt in length
    dt, time step in seconds
    Wn, window width, number of time steps'''
    event_didxs = (event_times/dt).astype(np.int64)
    dsr = np.zeros((len(event_times),Wn),dtype=np.bool)
    ii=0
    for didx in event_didxs:
        dsr[ii,:] = dst[didx:didx+Wn]
        ii+=1
    return dsr

@njit
def JSdiv(p,q):
    ''' JSDIV   Jensen-Shannon divergence.
      D = JSDIV(P,Q) calculates the Jensen-Shannon divergence of the two
      input distributions.'''
    # JS-divergence
    m = (p + q) / 2;
    D1 = KLdist(p,m)
    D2 = KLdist(q,m)
    return (D1 + D2) / 2

@njit
def KLdist(p,q):
    '''KLDIST   Kullbach-Leibler distance. 
    D = KLDIST(P,Q) calculates the Kullbach-Leibler distance
    (information divergence) of the two input distributions.'''
    p2 = p[p*q>0]
    q2 = q[p*q>0]
    p2 = p2/np.sum(p2) # renormalize
    q2 = q2/np.sum(q2)
    return np.sum(p2*np.log(p2/q2))

@njit
def makep_nmb(kld,kn):
    '''Calculates p value from distance matrix.'''
    pnhk = kld[0:kn-1,0:kn-1]
    f_pnhk = pnhk.flatten()
    # flatten fortran order
    for irow in range(pnhk.shape[0]):
        for icol in range(pnhk.shape[1]):
            f_pnhk[irow*pnhk.shape[1]+icol] = pnhk[irow,icol]
    nullhypkld = f_pnhk[~np.isnan(f_pnhk)]
    testkld = np.median(kld[0:kn-1,kn-1])     # value to test
    sno = len(nullhypkld[:])          # sample size for nullhyp. distribution
    p_value = len(np.where(nullhypkld>=testkld)[0]) / sno
    Idiff = testkld - np.median(nullhypkld)   # information difference between baseline and test latencies
    return (p_value, Idiff)
  
@njit
def SALTY(spt_baseline, spt_test,dt=0.001,wn=0.005):
    wn*=1000
    dt*=1000
    tno,st = spt_baseline.shape
    nmbn = round(wn/dt)
    edges = np.arange(-1,nmbn+1)
    nm = floor(st/nmbn)
    lsi = np.zeros((tno,nm),dtype = np.int64)
    slsi = np.copy(lsi)
    hlsi = np.zeros((nmbn+1,nm+1), dtype = np.int64)
    nhlsi = np.zeros(hlsi.shape, dtype = np.float64) # needs to hold normalized vector
    counter = 0
    for t in np.arange(0,st,nmbn): # loop through the baseline windows
        for k in np.arange(0,tno):
            cspt = spt_baseline[k,t:t+nmbn] # current baseline windown
            lsi[k,counter] = -1 if ~cspt.any() else cspt.argmax()
        slsi[:,counter] = np.sort(lsi[:,counter])
        hst,_ = np.histogram(slsi[:,counter],edges)
        hlsi[:,counter] = hst
        nhlsi[:,counter] = hlsi[:,counter] / np.sum(hlsi[:,counter])
        counter+=1
        
    # ISI histogram - test
    tno_test = spt_test.shape[0]   # number of trials
    lsi_tt = np.repeat(np.array(np.nan),tno_test)  # preallocate latency matrix
    for k in np.arange(0,tno_test):  # loop through trials
        cspt = spt_test[k,0:nmbn]   # current test window
        lsi_tt[k] = -1 if ~cspt.any() else cspt.argmax()

    slsi_tt = np.sort(lsi_tt)   # sorted latencies
    hst,_ = np.histogram(slsi_tt,edges)
    hlsi[:,counter] = hst # latency histogram
    nhlsi[:,counter] = hlsi[:,counter] / np.sum(hlsi[:,counter])   # normalized latency histogram

    # JS-divergence
    kn = nm + 1   # number of all windows (nm baseline win. + 1 test win.)
    jsd = np.reshape(np.repeat(np.nan,kn**2),(kn,-1))
    for k1 in range(kn):
        D1 = nhlsi[:,k1]  # 1st latency histogram
        for k2 in np.arange(k1+1,kn):
            D2 = nhlsi[:,k2]   # 2nd latency histogram
            jsd[k1,k2] = sqrt(JSdiv(D1,D2)*2)  # pairwise modified JS-divergence (real metric!)
    (p,I)=makep_nmb(jsd,kn)
    return (p,I)
