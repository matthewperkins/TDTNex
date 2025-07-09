import numba as nb
from numba import njit
@njit
def rolling_RR_guess(QRS_times,RRnominal,wlen=4):
    win_center = (wlen//2)-1
    indexs = np.arange(win_center,len(QRS_times)-(wlen-win_center-1))
    local_ratio = np.zeros(len(indexs),dtype=(np.float64))
    nominal_ratio = np.zeros(len(indexs),dtype=(np.float64))
    QRSpeaks_idxs = np.zeros(len(indexs),dtype=(np.int_))
    for ii,idx in enumerate(indexs):
        pre = np.mean(np.diff(QRS_times[idx-win_center:idx+1]))
        post = np.mean(np.diff(QRS_times[idx+1:idx+1+(wlen-win_center-1)]))
        local_ratio[ii] = np.diff(QRS_times[idx:idx+2])[0]/((pre+post)/2)
        nominal_ratio[ii] = np.diff(QRS_times[idx:idx+2])[0]/RRnominal
        QRSpeaks_idxs[ii]=idx
    return local_ratio,nominal_ratio,QRSpeaks_idxs   

# lets set this up with njit?
@nb.njit
def fill_missing_QRS(QRS_idxs,skip_idxs,prov_QRS_idxs,
                     xs,up_proms,envelope,no_choke=1000):
    add_idxs = np.zeros(len(skip_idxs),dtype=np.int64)
    add_order = np.zeros(len(skip_idxs),dtype=np.int64)
    num_add = 0
    # limit the range of the skip_idxs so that I dont run off the end of QRSup_idxs_keep
    print(skip_idxs[-1],len(QRS_idxs))
    if skip_idxs[-1]>len(QRS_idxs):
        print('len prob')
        print(skip_idxs[-1],QRS_idxs[-1])
        skip_idxs = skip_idxs[skip_idxs<len(QRS_idxs)-4]
        print(skip_idxs[-1],len(QRS_idxs))
    for ii,skip_idx in enumerate(skip_idxs):
        # take the preceeding and suceeding RR 
        # have to make sure I don't run off the end of the QRSup_idxs
        preRR = np.diff(xs[QRS_idxs][skip_idx-1:skip_idx+1])[0]
        skipRR = np.diff(xs[QRS_idxs][skip_idx:skip_idx+2])[0]
        postRR = np.diff(xs[QRS_idxs][skip_idx+1:skip_idx+3])[0]
        estRR = (preRR+postRR)/2
        ratio = skipRR/estRR
        _t = xs[QRS_idxs[skip_idx]]
        _l = _t+estRR*0.8
        _r = _t+estRR*1.2
        # reintroduce, conditional to the amplitude?
        slice_bounds = np.searchsorted(xs[prov_QRS_idxs],np.array([_l,_r]))
        cand_peaks = np.arange(slice_bounds[0],slice_bounds[1])
        if np.size(cand_peaks)==0:
            #print('no peaks at the appropriate times???')
            continue
        max_cand = np.argmax(QRSsig[prov_QRS_idxs[cand_peaks]])
        if up_proms[cand_peaks[max_cand]]<=envelope[cand_peaks[max_cand]]*0.1:
            #print("candidate %d too small, don't introduce" % prov_QRS_idxs[cand_peaks[max_cand]])
            continue
        else:
            reintro = prov_QRS_idxs[cand_peaks[max_cand]]
            add_idxs[num_add]=reintro
            add_order[num_add]=np.searchsorted(QRS_idxs,reintro)
            num_add+=1
        if ii>no_choke:
            break
    return (add_idxs[0:num_add],add_order[0:num_add],num_add)

@nb.njit()
def make_QRS_artifact_rolling(emg, art_idxs, win_len_dp,span = 100):
    deart = np.copy(emg)
    half_win_dp = int(win_len_dp//2)
    first_safe = 0
    for ii in range(len(art_idxs)):
        if art_idxs[ii]-half_win_dp<0:
            first_safe+=1
        else:
            break
    last_safe = len(art_idxs)-1
    for ii in range(len(art_idxs)-1):
        if art_idxs[last_safe]+half_win_dp>=len(emg):
            last_safe-=1
        else:
            break
    span_r = np.array([span//2,span//2],dtype=np.int64)
    print(last_safe,first_safe)
    art = np.zeros((last_safe-first_safe,win_len_dp),dtype=np.float64)
    art_count = np.arange(first_safe,last_safe,dtype=np.int64)
    for ii,art_idx in enumerate(art_idxs[first_safe:last_safe]):
        _start = ii-span_r[0]
        if _start<0:
            _start=0
        this_span = art_idxs[_start:ii+span_r[1]]
        safe_span = this_span[(this_span>=art_idxs[first_safe])&(this_span<art_idxs[last_safe])]
        for i in safe_span:
            art[ii]+=emg[i-half_win_dp:i-half_win_dp+win_len_dp]
        art[ii]=art[ii]/len(safe_span)
    for ii,art_idx in enumerate(art_idxs[first_safe:last_safe]):
        deart[art_idx-half_win_dp:art_idx-half_win_dp+win_len_dp]-=art[ii]
    return art,deart,art_count

# had some problems with heart rate contamination of resp during the 20% co2
@nb.njit
def drop_doublets(pks,trace,thresh):
    IBI_amp = np.zeros(len(pks)-1)
    for i in range(len(pks)-1):
        IBI_amp[i] = np.mean(trace[pks[i]:pks[i+1]])
    # > 1 emperically determined from histogram, 
    # not sure if there are triplets here so often.
    doublet_first = np.where(IBI_amp > thresh)[0]
    doublet_keep = np.zeros(doublet_first.shape,dtype=np.int64)
    doublet_drop = np.zeros(doublet_first.shape,dtype=np.int64)
    for i in range(len(doublet_first)):
        _fa = trace[pks[doublet_first[i]]]
        _sa = trace[pks[doublet_first[i]+1]]
        doublet_keep[i] = doublet_first[i] if _fa >= _sa else doublet_first[i]+1
        doublet_drop[i] = doublet_first[i] if _fa < _sa else doublet_first[i]+1
    return np.delete(pks,doublet_drop)
