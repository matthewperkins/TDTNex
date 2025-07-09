from neo import NeuroExplorerIO
#%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.transforms as transforms
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import mode
import os
import tdt
from pandas import IndexSlice as pidx
from numba import njit
import pandas as pd
from scipy.stats import zscore, mode, sem
from math import floor, sqrt


# maybe make a dataframe, row for each spike, 
## column indexs are wire sort code.
## values are EMGidx, pNeuidx, nextime, tdt_time, and a waveform array?
# read the data in wire order frome the tdt, for each time, calculate the nex time
# grouby by wire, iterate through the spiketrains for each wire, use timestamp as a key to assign the sort code
# wire 
# maybe make a dataframe, row for each spike, 
## column indexs are wire sort code.
## values are EMGidx, pNeuidx, nextime, tdt_time, and a waveform array?
# read the data in wire order frome the tdt, for each time, calculate the nex time
# grouby by wire, iterate through the spiketrains for each wire, use timestamp as a key to assign the sort code
# wire

@njit 
def trig_signal_avgsem(Signal,fs,peak_indexs,lpad,rpad):
    # use the bite burst times to make a triggered average
    dp_lpad = int(fs*lpad)
    dp_rpad = int(fs*rpad)
    nsamples = dp_lpad+dp_rpad
    sig_avg = np.zeros(nsamples)

        
    # # Create a vector from 0 up to nsamples
    start_indexs = peak_indexs-dp_lpad
    
    # do some bounds checking?
    # should do this with NP search sorted instead.
    # do I have to mask the indexs at all?
    _m =(peak_indexs+dp_rpad)<len(Signal)
    _m&=(peak_indexs-dp_lpad)>=0

    for i, start_index in enumerate(start_indexs[_m]):
        sig_avg+=Signal[start_index:start_index+nsamples]
    sig_avg = sig_avg/len(start_indexs)

    # loop through again to compute the sem
    print('here now')
    sig_sem = np.zeros(sig_avg.shape)
    for i, start_index in enumerate(start_indexs[_m]):
        sig_sem+=(Signal[start_index:start_index+nsamples]-sig_avg)**2
    sem_a = np.sqrt(sig_sem/len(start_indexs))
    avg_xs = np.linspace(-lpad,rpad,len(sig_avg))
    return avg_xs, sig_avg, sem_a

# implement run length encoding njit, to help deduplicate my toe-aligned threshold crossings
@njit
def rle(inarray):
        """ run length encoding. Partial credit to R rle function. 
            Multi datatype arrays catered for including non Numpy
            returns: tuple (runlengths, startpositions, values) """
        ia = np.asarray(inarray)                # force numpy
        n = len(ia)
        if n == 0: 
            return (None, None, None)
        else:
            y = ia[1:] != ia[:-1]               # pairwise unequal (string safe)
            i = np.append(np.where(y)[0], n - 1)   # must include last element posi
            z = np.diff(np.append(np.array([-1]), i))      # run lengths
            p = np.cumsum(np.append(0, z))[:-1] # positions
            return(z, p, ia[i])

def get_avg_fps_float(movie):
    import subprocess
    get_fps_command = ['ffprobe',
                       '-v',
                       'error',
                       '-select_streams', 
                       'v:0', 
                       '-show_entries', 
                       'stream=avg_frame_rate', 
                       '-of', 
                       'default=nokey=1:noprint_wrappers=1']
    get_fps_command+=[movie]
    rt = subprocess.run(get_fps_command,capture_output=True)
    return eval(rt.stdout)

def get_avg_fps_frac(movie):
    import subprocess
    get_fps_command = ['ffprobe',
                       '-v',
                       'error',
                       '-select_streams', 
                       'v:0', 
                       '-show_entries', 
                       'stream=avg_frame_rate', 
                       '-of', 
                       'default=nokey=1:noprint_wrappers=1']
    get_fps_command+=[movie]
    rt = subprocess.run(get_fps_command,capture_output=True)
    return rt.stdout.decode().rstrip()

def get_r_fps(movie):
    import subprocess
    get_fps_command = ['ffprobe',
                       '-v',
                       'error',
                       '-select_streams', 
                       'v:0', 
                       '-show_entries', 
                       'stream=r_frame_rate', 
                       '-of', 
                       'default=nokey=1:noprint_wrappers=1']
    get_fps_command+=[movie]
    rt = subprocess.run(get_fps_command,capture_output=True)
    return eval(rt.stdout)

def count_frames(movie):
    import subprocess
    import os
    assert os.path.exists(movie), "Path %s, does not seem to exist"
    count_frames_command = ['ffprobe',
                       '-v',
                       'error',
                       '-select_streams', 
                       'v:0', 
                       '-show_entries', 
                       'stream=nb_frames', 
                       '-of', 
                       'default=nokey=1:noprint_wrappers=1']
    count_frames_command+=[movie]
    rt = subprocess.run(count_frames_command,capture_output=True)
    return eval(rt.stdout)

def get_movie_dur(movie):
    import subprocess
    get_dur_command = ['ffprobe',
                       '-v',
                       'error',
                       '-show_entries',
                       'format=duration',
                       '-of',
                       'default=noprint_wrappers=1:nokey=1']
    get_dur_command+=[movie]
    rt=subprocess.run(get_dur_command,capture_output=True)
    return float(rt.stdout.decode().rstrip())

# lets dig out the seconds from the time stamps
def time_stamp_to_sec(ts):
    import re
    retimestamp = re.compile(r'.*(?P<ts_hour>[0-9]{2})[-_:](?P<ts_minute>[0-9]{2})[-_:](?P<ts_second>[0-9]{2})\.(?P<ts_millisec>[0-9]{3})')
    groupdict = retimestamp.search(ts)
    ts_h = float(groupdict['ts_hour'])
    ts_m = float(groupdict['ts_minute'])
    ts_s = float(groupdict['ts_second'])
    ts_ms = float(groupdict['ts_millisec'])
    # for some reason, ffmpeg has stopped recognizing the actual fps for the gstreamer grabbed streams, 
    # but I seem to get the correct fps with backend = cv2.CAP_GSTREAMER
    time_in_sec = ts_h*60*60+ts_m*60+ts_s+ts_ms/1000
    return time_in_sec

# for going back to the movie
def sec_to_time_stamp(sec,tdt_d,frame_epoc_name='FrmN'):
    from math import modf
    if frame_epoc_name is not None:
        sec-=tdt_d.epocs[frame_epoc_name].onset[0]
    hr,h = modf(sec/3600)
    mr,m = modf(hr*60)
    sr,s = modf(mr*60)
    ms = int(sr*1000)
    return ("%02d:%02d:%02d.%03d" % (h,m,s,ms))

def sec_to_ts(sec,frame_onests):
    from math import modf
    off_sec-=frame_onests
    hr,h = modf(off_sec/3600)
    mr,m = modf(hr*60)
    sr,s = modf(mr*60)
    ms = int(sr*1000)
    return ("%02d:%02d:%02d.%03d" % (h,m,s,ms))

@njit
def find_artifact_idxs(spiketimes,artifact_times,window = 0.0008):
    # preindex array
    spikes_to_drop = np.repeat(np.array([-1],dtype=np.int64),10000)
    counter = 0
    for artifact_time in artifact_times:
        artifact_spikes = np.where(np.abs(spiketimes - artifact_time)<window)[0]
        spikes_to_drop[counter:counter+len(artifact_spikes)]=artifact_spikes
        counter+=len(artifact_spikes)
    return np.sort(spikes_to_drop[0:counter])

@njit
def count_snips(event_times,unit_times,lpad,rpad):
        nsnips = int(0)
        for t in event_times:
            nsnips+=np.sum((unit_times>(t-lpad))&(unit_times<(t+rpad)))
        return nsnips

@njit 
def make_r_bins(bin_width,lpad,rpad):
    left_side_bins = np.arange(0,-lpad,-bin_width,dtype = np.float64)[::-1]
    right_side_bins = np.arange(bin_width,rpad,bin_width, dtype = np.float64)
    bins = np.zeros(len(left_side_bins)+len(right_side_bins))
    bins[0:len(left_side_bins)]=left_side_bins
    bins[len(left_side_bins):]=right_side_bins
    return bins

@njit
def make_raster(bin_width,nsnips,event_times,unit_times,lpad,rpad,waveforms):
    # make the bins
    bins = make_r_bins(bin_width,lpad,rpad)
    raster_segs = np.zeros((nsnips,30,2))
    # do the xs on the raster_segs collection just 0-30
    raster_segs[:,:,0]=np.arange(30)
    evntsArray = np.zeros((nsnips,))
    evnts = []
    rates = np.zeros((len(event_times),len(bins)-1))
    _seg_idx=0
    for ii,t in enumerate(event_times):
        _mask = (unit_times>(t-lpad))&(unit_times<(t+rpad))
        raster_segs[_seg_idx:_seg_idx+_mask.sum(),:,1]=waveforms[_mask,:]
        #evnts.append(g[_mask]['TDTts'].values-t) # subtract t shift to zero
        # apparently numba 0.45 handles lists njit, but must have strict homogenous types.
        evnts.append(unit_times[_mask]-t) # subtract t shift to zero
        #h,bx = np.histogram(g[_mask]['TDTts'].values-t,bins = bins)
        h,bx = np.histogram(unit_times[_mask]-t,bins = bins)
        rates[ii] = (h/bin_width)
        evntsArray[_seg_idx:_seg_idx+_mask.sum()]=evnts[-1]
        _seg_idx+=_mask.sum()
    return (evnts,evntsArray,raster_segs,(rates,bx))
    
class TDTNex(object):
    def __init__(self, tdt_file_path, nex_file_path):
        """For the alignment and manipulation of TDT data files with manually cluster cutted data from Offline sorter,
        that has been exported to the Neurodata Explorer format. Note should submit pull request to change waveform offset
        in NEO to fix waveform bug (is 2 should be 4). Not sure how widely this applies. ALSO, DO NOT INVALIDATE waveforms 
        when cluster cutting, this will fuck up aligning the records back to TDT. Finally, add a neareset CameraFrames column
        to the coordiinated unit dataframe. If the nearest frame is > 11 milliseconds away, is NA. Also, alignment of the files
        works by exact matching of time stamps, which is brittle, because there is some weird clock skew in the NEX file
        that comes out of OffLineSorter. This skew is measured, and the expected neotime stamps are created by multipyling
        the TDT time stamps by the a coefficent. These expected time stamps areout then rounded to 7 digits, and their
        matching timestamp is looked for on the appropriate wire from the NEX signals. Base on my experience the precision 
        of the rounding changes a bit from file to file, and may need to be tweaked"""
        self._tdt_fp = tdt_file_path
        self._nex_fp = nex_file_path
        self.tdt = tdt.read_block(self._tdt_fp)
        self.nex = NeuroExplorerIO(self._nex_fp)
        self._tdt_dur = self.tdt.info.duration.total_seconds()
        self._convolve_s = None # state stuff
        self._digRC = None # state stuff 
        self._masRC = None # state stuff
        try:
            self.EMG = self.tdt.streams.EMGx.data
        except AttributeError:
            self.EMG = None
        self.seg = self.nex.read_segment()
        # now have to deal with name differences in the default snip and streams names between synapse and openex.
        if 'eNeu' not in self.tdt.snips.keys():
            # see if eNe1 in keys
            if 'eNe1' in self.tdt.snips.keys():
                print('using eNe1 as snips name')
                self.tdt.snips.eNeu = self.tdt.snips.eNe1
            else:
                raise ValueError("snips name is no good.")
        if 'pNeu' not in self.tdt.streams.keys():
            # see if pNe1 in keys
            if 'pNe1' in self.tdt.streams.keys():
                print('using pNe1 as snips name')
                self.tdt.streams.pNeu = self.tdt.streams.pNe1
            else:
                raise ValueError("stream Neu name is no good.")
        self._make_event_df() # side effect to add df to self.
        self._make_NexSort_df() # side-effect function add df to self
        self._make_Unit_df() # side-effect function add df to self
        
    def _ts_pNeu_idx(self,ts):
        return(int(self.tdt.streams.pNeu.fs*ts))
    
    def _ts_EMGx_idx(self,ts):
        if self.EMG is None:
            return None
        return(int(self.tdt.streams.EMGx.fs*ts))
    
    def pNeu(self,start=None,stop=None):
        pNeu_dur = self.tdt.streams.pNeu.data.shape[1]/self.tdt.streams.pNeu.fs
        if start is not None:
            if (start>0)&(start<=pNeu_dur):
                S = start
            else:
                print('Start arg is bad, setting to 0')
                S = 0
        else:
            S = 0
        if stop is not None:
            if ((stop>0)&(stop>start)&(stop<pNeu_dur)):
                E = stop
            else:
                print('Stop arg is bad, setting to end of file')
                # I noticed that the end of the TDT record may not be accurately reflected
                # between different streams of data!!!
                # like some streams mayhave more points in them than others.
                # is better to compute XS of data based on
                #  number of points in the stream, and the fs
                E = pNeu_dur
                print(E,pNeu_dur)
        else:
            E = pNeu_dur
        Sidx,Eidx = self._ts_pNeu_idx(S),self._ts_pNeu_idx(E)
        data = self.tdt.streams.pNeu.data[:,Sidx:Eidx]
        xs = np.linspace(S,E,data.shape[1])
        return xs,data

    def EMGx(self,start=None,stop=None,ztrans=False):
        EMGx_dur = self.tdt.streams.EMGx.data.shape[1]/self.tdt.streams.EMGx.fs
        sig_mean = self.tdt.streams.EMGx.data.mean(axis=1)
        sig_std = self.tdt.streams.EMGx.data.std(axis=1)
        if start is not None:
            if (start>0)&(start<=EMGx_dur):
                S = start
            else:
                print('Start arg is bad, setting to 0')
                S = 0
        else:
            S = 0
        if stop is not None:
            if ((stop>0)&(stop>start)&(stop<EMGx_dur)):
                E = stop
            else:
                print('Stop arg is bad, setting to end of file')
                E = EMGx_dur
                print(E,EMGx_dur)
        else:
            E = EMGx_dur
        Sidx,Eidx = self._ts_EMGx_idx(S),self._ts_EMGx_idx(E)
        data = np.copy(self.tdt.streams.EMGx.data[:,Sidx:Eidx])
        if ztrans is True:
            for	ii,(sig_m,sig_std) in enumerate(list(zip(sig_mean,sig_std))):
                data[ii,:]=(data[ii,:]-sig_m)/sig_std
        xs = np.linspace(S,E,data.shape[1])
        return xs,data
                
    def _make_event_df(self):
        """Specific to epocs with offsets"""
        #calc the length of event df
        len_ev_df = np.array([len(v.onset) for k,v in self.tdt.epocs.items()]).sum().astype('int')
        print(len_ev_df)
        ev_df = pd.DataFrame({'name':['NA']*len_ev_df,
                              'onset':np.zeros((len_ev_df,),dtype=np.float64),
                              'offset':np.zeros((len_ev_df,),dtype=np.float64),
                              'data':np.zeros((len_ev_df,),dtype=np.float64)
                              })
        _idx = 0
        # because some of the offsets are not recorded in the tdt file,
        # I should specify explictly that some of these are not present and indicate when Infs are added.
        for k,v in self.tdt.epocs.items():
            tdt_ev = self.tdt.epocs[k]
            ev_df.loc[_idx:_idx+len(tdt_ev.onset)-1,'name'] = k
            ev_df.loc[_idx:_idx+len(tdt_ev.onset)-1,'onset'] = tdt_ev.onset
            ev_df.loc[_idx:_idx+len(tdt_ev.offset)-1,'offset'] = tdt_ev.offset
            ev_df.loc[_idx:_idx+len(tdt_ev.onset)-1,'data'] = tdt_ev.data
            # careful of singleton value onset epocs that don't end 
            _idx+=len(tdt_ev.onset)    
        self.ev_df = ev_df

    def _make_Unit_df(self):
        from pandas import IndexSlice as pidx
        tdt = self.tdt
        # first fill in the nex sortcode to the snips struct as a new attribute
        # tdt.snips.eNeu.nexsortcode
        tdt.snips.eNeu.nexsortcode = np.copy(tdt.snips.eNeu.sortcode)
        nxdf = self.nex_df.reset_index().copy()
        for wn,g in nxdf.groupby('wire'):
            sortcodes  = g.sort_values('st')['SC']
            # assign the sort codes to the wire keep in time order.
            tdt.snips.eNeu.nexsortcode[tdt.snips.eNeu.chan==wn] = g.sort_values('st')['SC']

        # now create a dataframe for the spikes    
        frlen = len(tdt.snips.eNeu.ts)
        unitdf = pd.DataFrame({'wire':np.zeros((frlen,),dtype=np.int64),
                            'TankSC':np.zeros((frlen,),dtype=np.int64), # use -1 for unsorted
                            'NEXSC':np.zeros((frlen,),dtype=np.int64), # use -1 for unsorted
                            'TDTts':np.zeros((frlen,),dtype=np.float64),
                            'TDTwvidx':np.zeros((frlen,),dtype=np.int64),
                            'EMGidx':np.zeros((frlen,),dtype=np.int64),
                            'pNeuidx':np.zeros((frlen,),dtype=np.int64)})


        # fill in the TDTts and NEOts by wire
        _idx_offset = 0
        pNeufs = tdt.streams.pNeu.fs
        # if there is no EMGx just leave as zeros
        if self.EMG is not None:
            EMGfs = tdt.streams.EMGx.fs
        for wire in np.unique(self.tdt.snips.eNeu.chan.flatten()):
            _wt = tdt.snips.eNeu.ts[np.argwhere(tdt.snips.eNeu.chan.flatten()==wire).flatten()].flatten()
            _nexsc = tdt.snips.eNeu.nexsortcode[np.argwhere(tdt.snips.eNeu.chan.flatten()==wire).flatten()].flatten()
            _tanksc = tdt.snips.eNeu.sortcode[np.argwhere(tdt.snips.eNeu.chan.flatten()==wire).flatten()].flatten()
            _wwvidx = np.argwhere(tdt.snips.eNeu.chan.flatten()==wire).flatten()
            unitdf.loc[_idx_offset:_idx_offset+len(_wt)-1,'wire']=wire
            unitdf.loc[_idx_offset:_idx_offset+len(_wt)-1,'TankSC']=_tanksc
            unitdf.loc[_idx_offset:_idx_offset+len(_wt)-1,'NEXSC']=_nexsc
            unitdf.loc[_idx_offset:_idx_offset+len(_wt)-1,'TDTts']=_wt
            unitdf.loc[_idx_offset:_idx_offset+len(_wt)-1,'TDTwvidx']=_wwvidx
            if self.EMG is not None:
                EMGfs = tdt.streams.EMGx.fs
                unitdf.loc[_idx_offset:_idx_offset+len(_wt)-1,'EMGidx']=(_wt*EMGfs).astype(int)
            unitdf.loc[_idx_offset:_idx_offset+len(_wt)-1,'pNeuidx']=(_wt*pNeufs).astype(int)
            #
            _idx_offset+=len(_wt)

        unitdf.set_index(['wire'],inplace=True)
        # now get the waveforms
        waveforms = {}
        for (wn,nexsc),g in unitdf.groupby(['wire','NEXSC']):
            # need to pull out the waves here.
            _wvs = self.tdt.snips.eNeu.data[g.TDTwvidx.values]
            waveforms[(wn,nexsc)]=np.copy(_wvs)
        self.unitdf = unitdf.reset_index().set_index(['wire','NEXSC']).sort_index().copy()
        # coount the number of sorted units, i.e. SC not zero:
        nunits = 0
        for (wire, sc),g in self.unitdf.groupby(['wire','NEXSC']):
            if sc==0:
                continue
            else:
                nunits+=1
        self.nunits = nunits
        self.waveforms = waveforms

    def drop_artifacts(self, Times, window = 0.0008):
        #def drop_opto_artifacts(UnitDf, Times, window=0.0008):
        drop_ilocs = find_artifact_idxs(self.unitdf.TDTts.values, Times, window=window)
        if np.size(drop_ilocs)!=0:
            print("dropping %d spikes" % np.size(drop_ilocs))
        keep_ilocs = np.setdiff1d(np.arange(len(self.unitdf)),drop_ilocs)
        print("len unitdf pre drop %d" % len(self.unitdf))
        dropped = self.unitdf.iloc[drop_ilocs]
        self.unitdf = self.unitdf.iloc[keep_ilocs]
        print("len unitdf post drop %d" % len(self.unitdf))
        self.unitdf = self.unitdf.reset_index().set_index(['wire','NEXSC']).sort_index().copy()
        len_wv = 0
        for k in self.waveforms:
            len_wv+=len(self.waveforms[k])
        print("len waveforms pre drop %d" % len_wv)

        # have to drop the waveforms too, just pull them in from the tdt object again:
        waveforms = {}
        for (wn,nexsc),g in self.unitdf.groupby(['wire','NEXSC']):
            # need to pull out the waves here.
            _wvs = self.tdt.snips.eNeu.data[g.TDTwvidx.values]
            waveforms[(wn,nexsc)]=np.copy(_wvs)
        self.waveforms = waveforms
        for k in self.waveforms:
            len_wv+=len(self.waveforms[k])
        print("len waveforms post drop %d" % len_wv)
        self.unitdf = self.unitdf.reset_index().set_index(['wire','NEXSC']).sort_index().copy()
        return dropped

    def _make_NexSort_df(self):
        tdt = self.tdt
        # do the sort codes as integers, make unsorted = 0
        from string import ascii_lowercase
        SCdict = {ltr:SC+1 for SC,ltr in enumerate([x for x in ascii_lowercase[0:26]])}
        SCdict['U']=0

        # make a data frame from all the nexsorted stuff
        # first collect the real spike trains, (ending in _wf), 
        # zip them with their spiketrain_index for waveform fetching
        # real spike trains are those signal with '_wf' suffix
        real_spktrns = []
        for st_num, st in enumerate(self.seg.spiketrains):
            if st.name[-2:]!='wf':
                continue
            else:
                real_spktrns.append((st_num,st))

        n_spktrn = len(real_spktrns)

        # also count the total number of spikes for sorting
        nNexSpikes = np.array([len(st) for _,st in real_spktrns]).sum()
        # as a sanity check this should be the same as the number of tdt_d spike
        assert nNexSpikes==len(tdt.snips.eNeu.ts),"num spikes in NexFile % is different from in TDT file (%d)" % (nNexSpikes, len(tdt.snips.eNeu.ts))

        # spike train data frame: wire, sort_code, st_num with in segment
        NexSorted_df = pd.DataFrame({'wire':np.zeros((nNexSpikes,),dtype=np.int64),
                                     'SC':np.zeros((nNexSpikes,),dtype=np.int64,),
                                     'st_num':np.zeros((nNexSpikes,),dtype=np.int64,),
                                     'st':np.zeros((nNexSpikes,),dtype=np.float64,)})
        _idx = 0
        for st_num, st in real_spktrns:
            wire = int(st.name[3:5])
            NexSorted_df.loc[_idx:_idx+len(st)-1,'wire']=wire
            SC = SCdict[st.name[5]]
            NexSorted_df.loc[_idx:_idx+len(st)-1,'SC']=SC
            NexSorted_df.loc[_idx:_idx+len(st)-1,'st_num']=st_num
            NexSorted_df.loc[_idx:_idx+len(st)-1,'st']=st.times.magnitude
            _idx+=len(st)
        NexSorted_df.set_index(['wire','SC'],inplace=True)
        self.nex_df = NexSorted_df
        
    def UnitRaster(self,wire,sc,event_times,lpad,rpad,bin_width=None):
        """Return a list of events, an array of all events, and set of waveform segments
        """
        # clean up the signature to pass through to numba function
        unit_times = self.unitdf.loc[(wire,sc),'TDTts'].values
        waveforms = self.waveforms[(wire,sc)]
        assert(type(event_times)==type(np.array([]))),"run your shit Matt, make it an array"
        if bin_width is None:
            bin_width = (lpad+rpad)/40
        # count the snips, do this in a separate njit func, because njit cant type python's None
        nsnips = count_snips(event_times,unit_times,lpad,rpad)
        if nsnips<1:
            print("fewer than 1 snips")
            return (None,None,None,(None,None))
        evnts,evntsArray,raster_segs,(rates,bx) = make_raster(bin_width,nsnips,event_times,unit_times,lpad,rpad,waveforms)
        return evnts,evntsArray,raster_segs,(rates,bx)

    def PlotUnitRaster(self,wire,sc,times,lpad,rpad,hist=True,
                       time_offsets = None,
                       time_preceeds = None,
                       bin_width=0.1,hist_yscale=None, 
                       lwds=1,lineoff=1,linelen=1,
                       inset_yscale=None,raster_color='black',
                       plt_rand=False,addLabel = True,wv_lw = 0.25):
        evnts, evntsArray,raster_segs,(rates,bx) = self.UnitRaster(wire,sc,times,lpad,rpad)
        if evnts is None:
            print("no snips")
            return None, (None, None, None), (None, None)
        f,(hist_ax,raster_ax) = plt.subplots(2,1,sharex='all')
        raster_ax.set_position([0.15,0.15,0.6,0.6])
        hist_ax.set_position([0.15,0.75,0.6,0.25])
        # try sharing the x axis of the raster and the histogram
        # raster_ax.get_shared_x_axes().join(hist_ax, raster_ax)
        # hist_ax.set_xticklabels([])
        # then add the waveform axes
        wf_ax = plt.axes([0.75,0.75,0.25,0.25])
        # preindex a segs array for the random line collection
        totsnips = len(self.unitdf.loc[(wire,sc),'TDTts'])
        if totsnips>50:
            random_segs = np.zeros((50,30,2))
            random_segs[:,:,1] = self.waveforms[(wire,sc)][np.random.randint(0,totsnips-1,50)]
        else:
            random_segs = np.zeros((totsnips,30,2))
            random_segs[:,:,1] = self.waveforms[(wire,sc)][:]
        random_segs[:,:,0]=np.r_[0:30]
        raster_ax.eventplot(evnts,linewidths = lwds, linelengths = linelen, 
                            lineoffsets = lineoff, color = 'black')
        # now would like to add a patch if there are event offsets
        if time_offsets is not None and time_preceeds is not None:
            assert(len(times)==len(time_offsets)),"Length of Time Offsets %d, is different than that of Times %d" % (len(times),len(time_offsets))
            assert(len(times)==len(time_preceeds)),"Length of Time Preceeds %d, is different than that of Times %d" % (len(times),len(time_preceeds))
            from matplotlib.patches import Rectangle
            # this maybe slow for > 200 events
            for _i, (pre,_t,off) in enumerate(zip(time_preceeds,times,time_offsets)):
                _r= Rectangle((pre-_t,(lineoff*_i)-linelen/2),off-pre,lineoff,
                              color = 'blue',alpha = 0.6, ec = 'None')
                raster_ax.add_patch(_r)
        elif time_offsets is not None:
            assert(len(times)==len(time_offsets)),"Length of Time Offsets %d, is different than that of Times %d" % (len(times),len(time_offsets))
            from matplotlib.patches import Rectangle
            # this maybe slow for > 200 events
            for _i, (_t,off) in enumerate(zip(times,time_offsets)):
                _r= Rectangle((0,(lineoff*_i)-linelen/2),off-_t,lineoff,
                              color = 'blue',alpha = 0.6, ec = 'None')
                raster_ax.add_patch(_r)
                                                                                                                     
        # have to do the inset axes, histogram
        wf_ax.patch.set_alpha(0.02)
        raster_snips = LineCollection(raster_segs, linewidths=wv_lw,
                                colors=raster_color, 
                                linestyle='solid')
        rand_snips = LineCollection(random_segs, linewidths=wv_lw,
                                colors='blue', 
                                linestyle='solid')
        if plt_rand:
            wf_ax.add_collection(rand_snips)
        wf_ax.add_collection(raster_snips)
        wf_ax.set_xlim(0,30)
        if inset_yscale is None:
            wf_ax.set_ylim(min(raster_segs[:,:,1].flatten()),max(raster_segs[:,:,1].flatten()))
        else:
            wf_ax.set_ylim(*inset_yscale)
        wf_ax.xaxis.set_visible(False)
        wf_ax.yaxis.set_visible(False)
        bins = make_r_bins(bin_width,lpad,rpad)
        bh,bx = np.histogram(evntsArray,bins = bins)
        hist_ax.bar(bx[0:-1],bh/len(times)/bin_width,width = bin_width, align='edge')
        hist_ax.set_ylabel("inst. freq Hz, %.2f" % bin_width)
        hist_ax.xaxis.set_visible(False)
        hist_ax.set_xlim(-lpad,rpad)
        if hist_yscale is not None:
            hist_ax.set_ylim(*hist_yscale)
        raster_ax.set_xlim(-lpad,rpad)
        raster_ax.set_xlabel("time (s)")
        raster_ax.set_ylabel("trail num.")
        if addLabel:
            f.text(0.1,0.85,"w:%s,sc:%s" % (wire,sc),transform = f.transFigure)
        f.set_size_inches(4,4)
        return f, (hist_ax, raster_ax, wf_ax), (bh/bin_width/len(times), bx)
            
    def AllUnitRasters(self,times,lpad,rpad,hist=True,
                       time_offsets=None, bin_width = 0.1,fndec=None,
                       hist_yscale=None, lwds = 1, lineoff = 0.8,linelen = 0.8,
                       inset_yscale=None, raster_color='black',fntitle=False,frmt='png',plt_dir=None):
        # use TDT time, all in seconds
        if plt_dir is None:
            plt_dir = os.path.join(os.curdir, "Rasters")
        os.makedirs(plt_dir,exist_ok=True)
        for (wire, sc),g  in self.unitdf.groupby(['wire','NEXSC']):
            if sc==0:
                continue
            f,(hist_ax,raster_ax,wf_ax), (u_freq, bx) = self.PlotUnitRaster(
                wire,sc,times,lpad,rpad,
                time_offsets=time_offsets,
                hist=hist,bin_width=bin_width,
                hist_yscale=hist_yscale,
                lwds=lwds,lineoff=lineoff,
                linelen=linelen,inset_yscale=inset_yscale,
                raster_color=raster_color)
            if f is None:
                continue
            if fndec is None:
                f.savefig(os.path.join(plt_dir,"Raster_wire%02d_sc%s.%s" % (wire,sc,frmt)),
                        dpi = 300,transparent=True)
            else:
                f.savefig(os.path.join(plt_dir,"Raster_%s_wire%02d_sc%s.%s" % (fndec,wire,sc,frmt)),
                        dpi = 300,transparent=True)
            plt.close()
        
    def UnitPanel(self,nsnips=50,lattice=True):
        from math import sqrt, ceil
        # use the nunit count peformed during the unitdf construction.
        # just make a square of axes
        nrow = ceil(sqrt(self.nunits))
        f,axar = plt.subplots(nrow,nrow,sharex='all')
        if lattice:
            f,axar = plt.subplots(4,4,sharex='all')
            sc_cmap = plt.get_cmap('Set1')
            # compute global min and max for all units on a wire
            wire_ylims = {wire:[0,0] for wire in np.r_[1:17]}
            for (wire, sc),g in self.unitdf.groupby(['wire','NEXSC']):
                totsnips = len(g.TDTts)
                if totsnips<50:
                    tmp_wvs = self.waveforms[(wire,sc)][np.random.randint(0,totsnips-1,50)]
                else:
                    tmp_wvs = self.waveforms[(wire,sc)]
                print(tmp_wvs.flatten().shape)
                if min(tmp_wvs.flatten())<wire_ylims[wire][0]: wire_ylims[wire][0] = min(tmp_wvs.flatten()) 
                if max(tmp_wvs.flatten())>wire_ylims[wire][1]: wire_ylims[wire][1] = max(tmp_wvs.flatten()) 
        _unit_cnt = 0
        for (wire, sc),g in self.unitdf.groupby(['wire','NEXSC']):
            if sc==0:
                continue
            totsnips = len(g.TDTts)
            if totsnips>nsnips:
                random_segs = np.zeros((nsnips,30,2))
                random_segs[:,:,1] = self.waveforms[(wire,sc)][np.random.randint(0,totsnips-1,50)]
            else:
                random_segs = np.zeros((totsnips,30,2))
                random_segs[:,:,1] = self.waveforms[(wire,sc)][:]
            random_segs[:,:,0]=np.r_[0:30]
            if lattice:
                snip_color = sc_cmap(sc/10)
            else:
                snip_color = 'black'
            rand_snips = LineCollection(random_segs, linewidths=0.25,
                                   colors=snip_color,
                                   linestyle='solid')
            if lattice:
                ax = axar.flatten()[wire-1]
            else:
                ax = axar.flatten()[_unit_cnt]
            
            ax.add_collection(rand_snips)
            if lattice:
                ax.set_ylim(wire_ylims[wire])
            else:
                ax.set_ylim(min(random_segs[:,:,1].flatten()),
                            max(random_segs[:,:,1].flatten()))
            ax.text(0.65,0,"W:%d,SC:%d" % (wire,sc), transform = ax.transAxes, size = 8)
            _unit_cnt+=1
        [ax.set_xlim(0,30) for ax in axar.flatten()]
        f.suptitle("units for %s" % os.path.basename(self._nex_fp))
        f.set_size_inches(10,10)
        return f

    def GetWaves(self,wire,sc,start,stop,maxnwvs='all'):
        _unitdf = self.unitdf.reset_index().set_index(['wire','NEXSC'])
        g = _unitdf.loc[(wire,sc)]
        times = (start,stop)
        g_mask = g.TDTts.between(*times)
        n_wvs = g_mask.sum()
        if n_wvs==0:
            return None
        else:
            if maxnwvs=='all':
                return self.waveforms[(wire,sc)][g_mask,:]
            else:
                if maxnwvs>n_wvs:
                    maxnwvs=n_wvs
                _slct = np.random.randint(0,n_wvs-1,maxnwvs)
                return self.waveforms[(wire,sc)][g_mask,:][_slct]

    def SpikeTriggeredEMG(self, wire, sortcode,
                          MaxN=50000,
                          DigaChan=1,MastChan=2,
                          lpad=0.2,rpad=0.2, 
                          time_buckets = None,
                          plt_stderr=True, ylim=False,
                          convolve_s = None,pltdir = '.',**kwargs):
        """"""
        if 'EMGpltargs' in kwargs.keys():
            EMG_plt_args = kwargs['EMGpltargs']
        else:
            EMG_plt_args = {'digastric':{'color':'black'},
                            'maseter':{'color':'red'}}
        fs = self.tdt.streams.EMGx.fs
        # raw data option
        if convolve_s is None:        
            digRC = zscore(self.EMG[DigaChan,:])
            masRC = zscore(self.EMG[MastChan,:])
        else:
            if convolve_s == self._convolve_s:
                # cache of previous computation, sloppy
                digRC = self._digRC
                masRC = self._masRC
            else:
                digRC = zscore(np.convolve(np.abs(self.EMG[DigaChan,:]),
                               np.ones((int(convolve_s*fs),))/int(convolve_s*fs),mode = 'same'))
                masRC = zscore(np.convolve(np.abs(self.EMG[MastChan,:]),
                                 np.ones((int(convolve_s*fs),))/int(convolve_s*fs),mode = 'same'))
                self._digRC = digRC
                self._masRC = masRC
                self._convolve_s = convolve_s
        os.makedirs(pltdir, exist_ok=True)

        dp_lpad = self._ts_EMGx_idx(lpad)
        dp_rpad = self._ts_EMGx_idx(rpad)

        # allow to select some spikes for restricted time buckets, i.e. only spike 15 seconds after taste exposure.
        g = self.unitdf.reset_index().groupby(['wire','NEXSC']).get_group((wire,sortcode))
        # if there are no item buckets, set iter to all rows.
        if time_buckets is None:
            # have to drop spikes at beginning and end of the recording that I can not average
            times = g['TDTts'].values
            times = times[(times>lpad*1.1)&(times<self._tdt_dur-rpad*1.1)]
            print("w:%02d sc:%d, %d spikes for averaging" % (wire,sortcode,len(g)))
            if len(g)<3:
                print("too few spikes %d %d" % (wire,sortcode))
                return None
            nAvg = len(g)
            f,ax = plt.subplots(1,1)
        else:
            _mask = np.zeros((len(g),),dtype = np.bool)
            for S,E in time_buckets:
                _mask = _mask | g.TDTts.between(S,E)
            # if there are no spikes in any of the buckets, give up return none:
            nAvg = _mask.sum()
            if _mask.sum()<3:
                print("too few spikes %d %d" % (wire,sortcode))
                return None
            print("w:%02d sc:%d, %d spikes for averaging" % (wire,sortcode,nAvg))
            times = g.loc[_mask,'TDTts'].values
            times = times[(times>lpad)&(times<self._tdt_dur-rpad)]
            f,ax = plt.subplots(1,1)

        nsamples = dp_lpad+dp_rpad

        # # Create a vector from 0 up to nsamples
        sample_idx = np.arange(nsamples)


        # # Calculate the index of the first sample for each chunk
        # # Require integers, because it will be used for indexing

        ## drop spike times that I will not be able to average around 
        ## (i.e. clipped by beginning and end of file)
        start_idx = ((times - lpad) * fs).astype(int)
        start_idx = start_idx[(start_idx+nsamples)<len(digRC)-1]

        # # Use broadcasting to create an array with indices
        # # Each row contains consecutive indices for each chunk
        idx = start_idx[:, None] + sample_idx[None, :]
        
        # # Get all the chunks using fancy indexing
        print(idx.shape)
        # hey
        dig_ar = digRC[idx]
        mas_ar = masRC[idx]

        ax.plot(np.linspace(-lpad,rpad,nsamples),dig_ar.mean(axis=0),label = 'digastric',**EMG_plt_args['digastric'])
        ax.plot(np.linspace(-lpad,rpad,nsamples),mas_ar.mean(axis=0),label = 'maseter',**EMG_plt_args['maseter'])
        if plt_stderr:
            ax.fill_between(np.linspace(-lpad,rpad,nsamples),
                            dig_ar.mean(axis=0)-sem(dig_ar,axis=0),
                            dig_ar.mean(axis=0)+sem(dig_ar,axis=0),
                            alpha = 0.4,
                            **EMG_plt_args['digastric'])
            ax.fill_between(np.linspace(-lpad,rpad,nsamples),
                            mas_ar.mean(axis=0)-sem(mas_ar,axis=0),
                            mas_ar.mean(axis=0)+sem(mas_ar,axis=0),
                            alpha = 0.4,
                            **EMG_plt_args['maseter'])
        if ylim:
            ax.set_ylim(ylim)

        axins = plt.axes([0.75,0.75,0.2,0.2])
        f.add_axes(axins)
        axins.patch.set_alpha(0.02)
        # want to select waves for the inset axis that are from the period of time depicted in the raster
        # plot at most 50 waves, less if there are fewer spikes
        all_wvs = self.waveforms[(wire,sortcode)]
        num_rnd_wvs = len(all_wvs)-1 if (len(all_wvs)<50) else 50
        # plot a random selection from all the waves
        rnd_wvs = all_wvs[np.random.randint(0,len(all_wvs),num_rnd_wvs),:]
        rnd_segs = np.zeros(rnd_wvs.shape+(2,))
        rnd_segs[:,:,1] = rnd_wvs
        rnd_segs[:,:,0] = np.r_[0:30]
        rnd_snips = LineCollection(rnd_segs, linewidths=0.25,
                                    colors='blue', linestyle='solid')
        axins.add_collection(rnd_snips)
        if time_buckets is not None:
            # plot the raster waves
            raster_wvs = self.waveforms[(wire,sortcode)][_mask]
            raster_segs = np.zeros(raster_wvs.shape + (2,))
            raster_segs[:,:,1] = raster_wvs
            raster_segs[:,:,0] = np.r_[0:30]
            raster_snips = LineCollection(raster_segs, linewidths=0.25,
                                            colors='black', linestyle='solid')
            axins.add_collection(raster_snips)
        axins.set_zorder(10)
        ymin, ymax = np.min(rnd_segs[:,:,1]),np.max(rnd_segs[:,:,1])
        xmin, xmax = np.min(rnd_segs[:,:,0]),np.max(rnd_segs[:,:,0])
        axins.set_ylim(ymin,ymax)
        axins.set_xlim(xmin,xmax)
        axins.text(0,0.95,"w%02sc%d" % (wire,sortcode),transform = axins.transAxes)
        [x.set_visible(False) for x in [axins.xaxis, axins.yaxis]]

        f.suptitle("w%02sc%d,N=%d" % (wire,sortcode,nAvg))
        f.savefig(os.path.join(pltdir,"SpikeTriggeredEMG_Wire%02dSC%d.png" % (wire,sortcode)),
                  dpi = 300,transparent=True)
        return f

    def SpikeTriggeredStream(self, wire, sortcode,
                             StreamName, StreamIdx=None,
                             MaxN=50000,
                             lpad=0.2,rpad=0.2, 
                             time_buckets = None,
                             plt_stderr=True, ylim=False,
                             convolve_s = None,pltdir = '.',**kwargs):
        """"""
        if 'plt_args' in kwargs.keys():
            plt_args = kwargs['plt_args']
        else:
            plt_args = {'color':'black'}

        fs = self.tdt.streams[StreamName].fs
        # check the dimensionality of the stream
        if len(self.tdt.streams[StreamName].data.shape)==1:
            data = self.tdt.streams[StreamName].data
            # set the stream Idx to 0, for ease formating output later
            StreamIdx=0
        elif len(self.tdt.streams[StreamName].data.shape)==2:
            assert(StreamIdx is not None),"Stream has %d dim, must give an index to one dim" % self.tdt.streams[StreamName].data.shape[0]
            data = self.tdt.streams[StreamName].data[StreamIdx,:]
        # raw data option
        if convolve_s is None:        
            dataRC = zscore(data)
        else:
            dataRC = zscore(np.convolve(np.abs(data),
                            np.ones((int(convolve_s*fs),))/int(convolve_s*fs),mode = 'same'))
        os.makedirs(pltdir, exist_ok=True)

        dp_lpad = int(lpad*fs)
        dp_rpad = int(rpad*fs)
        nsamples = dp_lpad+dp_rpad

        # allow to select some spikes for restricted time buckets, i.e. only spike 15 seconds after taste exposure.
        g = self.unitdf.reset_index().groupby(['wire','NEXSC']).get_group((wire,sortcode))
        # if there are no item buckets, set iter to all rows.
        if time_buckets is None:
            # have to drop spikes at beginning and end of the recording that I can not average
            times = g['TDTts'].values
            times = times[(times>lpad*1.1)&(times<self._tdt_dur-rpad*1.1)]
            print("w:%02d sc:%d, %d spikes for averaging" % (wire,sortcode,len(g)))
            if len(g)<3:
                print("too few spikes %d %d" % (wire,sortcode))
                return None
            nAvg = len(g)
            f,ax = plt.subplots(1,1)
        else:
            _mask = np.zeros((len(g),),dtype = np.bool)
            for S,E in time_buckets:
                _mask = _mask | g.TDTts.between(S,E)
            # if there are no spikes in any of the buckets, give up return none:
            nAvg = _mask.sum()
            if _mask.sum()<3:
                print("too few spikes %d %d" % (wire,sortcode))
                return None
            print("w:%02d sc:%d, %d spikes for averaging" % (wire,sortcode,nAvg))
            times = g.loc[_mask,'TDTts'].values
            times = times[(times>lpad)&(times<self._tdt_dur-rpad)]
            f,ax = plt.subplots(1,1)

        # if there are a ton of spikes, randomly select times upto MaxN
        if len(times)>MaxN:
            from numpy.random import default_rng 

            rng = default_rng()
            times = rng.choice(times,MaxN,replace=False)

        # # Create a vector from 0 up to nsamples
        sample_idx = np.arange(nsamples)

        # # Calculate the index of the first sample for each chunk
        # # Require integers, because it will be used for indexing

        ## drop spike times that I will not be able to average around 
        ## (i.e. clipped by beginning and end of file)
        start_idx = ((times - lpad) * fs).astype(int)
        start_idx = start_idx[(start_idx+nsamples)<len(dataRC)-1]

        # # Use broadcasting to create an array with indices
        # # Each row contains consecutive indices for each chunk
        idx = start_idx[:, None] + sample_idx[None, :]
        
        # # Get all the chunks using fancy indexing
        print(idx.shape)
        # hey
        data_ar = dataRC[idx]

        ax.plot(np.linspace(-lpad,rpad,nsamples),data_ar.mean(axis=0),
                label = "%s %d" % (StreamName,StreamIdx),**plt_args)
        if plt_stderr:
            ax.fill_between(np.linspace(-lpad,rpad,nsamples),
                            data_ar.mean(axis=0)-sem(data_ar,axis=0),
                            data_ar.mean(axis=0)+sem(data_ar,axis=0),
                            alpha = 0.4,**plt_args)
        if ylim:
            ax.set_ylim(ylim)

        axins = plt.axes([0.75,0.75,0.2,0.2])
        f.add_axes(axins)
        axins.patch.set_alpha(0.02)
        # want to select waves for the inset axis that are from the period of time depicted in the raster
        # plot at most 50 waves, less if there are fewer spikes
        all_wvs = self.waveforms[(wire,sortcode)]
        num_rnd_wvs = len(all_wvs)-1 if (len(all_wvs)<50) else 50
        # plot a random selection from all the waves
        rnd_wvs = all_wvs[np.random.randint(0,len(all_wvs),num_rnd_wvs),:]
        rnd_segs = np.zeros(rnd_wvs.shape+(2,))
        rnd_segs[:,:,1] = rnd_wvs
        rnd_segs[:,:,0] = np.r_[0:30]
        rnd_snips = LineCollection(rnd_segs, linewidths=0.25,
                                    colors='blue', linestyle='solid')
        axins.add_collection(rnd_snips)
        if time_buckets is not None:
            # plot the raster waves
            raster_wvs = self.waveforms[(wire,sortcode)][_mask]
            raster_segs = np.zeros(raster_wvs.shape + (2,))
            raster_segs[:,:,1] = raster_wvs
            raster_segs[:,:,0] = np.r_[0:30]
            raster_snips = LineCollection(raster_segs, linewidths=0.25,
                                            colors='black', linestyle='solid')
            axins.add_collection(raster_snips)
        axins.set_zorder(10)
        ymin, ymax = np.min(rnd_segs[:,:,1]),np.max(rnd_segs[:,:,1])
        xmin, xmax = np.min(rnd_segs[:,:,0]),np.max(rnd_segs[:,:,0])
        axins.set_ylim(ymin,ymax)
        axins.set_xlim(xmin,xmax)
        axins.text(0,0.95,"w%02sc%d" % (wire,sortcode),transform = axins.transAxes)
        [x.set_visible(False) for x in [axins.xaxis, axins.yaxis]]

        f.suptitle("w%02sc%d,N=%d,subsample=%d" % (wire,sortcode,nAvg,len(times)))
        f.savefig(os.path.join(pltdir,"SpikeTriggered_%s_Idx%d_%02dSC%d.png" % (StreamName, StreamIdx,wire,sortcode)),
                  dpi = 300,transparent=True)
        return f

    def EventTriggeredStream(self, EventTimes, EventName,
                             StreamName, StreamIdx=None,
                             lpad=0.2,rpad=0.2, 
                             time_buckets = None,
                             plt_stderr=True, ylim=False,
                             convolve_s = None,pltdir = '.',**kwargs):
        """"""
        fs = self.tdt.streams[StreamName].fs
        # check the dimensionality of the stream
        if len(self.tdt.streams[StreamName].data.shape)==1:
            data = self.tdt.streams[StreamName].data
            # set the stream Idx to 0, for ease formating output later
            StreamIdx=0
        elif len(self.tdt.streams[StreamName].data.shape)==2:
            assert(StreamIdx is not None),"Stream has %d dim, must give an index to one dim" % self.tdt.streams[StreamName].data.shape[0]
            data = self.tdt.streams[StreamName].data[StreamIdx,:]
        # raw data option
        if convolve_s is None:        
            dataRC = zscore(data)
        else:
            dataRC = zscore(np.convolve(np.abs(data),
                            np.ones((int(convolve_s*fs),))/int(convolve_s*fs),mode = 'same'))
        # plot kwargs?
        if 'plt_args' not in kwargs.keys():
            plt_args = {}
        else:
            plt_args = kwargs['plt_args']
        os.makedirs(pltdir, exist_ok=True)
        dp_lpad = int(lpad*fs)
        dp_rpad = int(rpad*fs)
        nAvg = len(EventTimes)
        f,ax = plt.subplots(1,1)
        nsamples = dp_lpad+dp_rpad

        # # Create a vector from 0 up to nsamples
        sample_idx = np.arange(nsamples)

        # # Calculate the index of the first sample for each chunk
        # # Require integers, because it will be used for indexing

        ## drop spike times that I will not be able to average around 
        ## (i.e. clipped by beginning and end of file)
        start_idx = ((EventTimes - lpad) * fs).astype(int)
        start_idx = start_idx[(start_idx+nsamples)<len(dataRC)-1]

        # # Use broadcasting to create an array with indices
        # # Each row contains consecutive indices for each chunk
        idx = start_idx[:, None] + sample_idx[None, :]
        
        # # Get all the chunks using fancy indexing
        print(idx.shape)
        data_ar = dataRC[idx]
        ax.plot(np.linspace(-lpad,rpad,nsamples),data_ar.mean(axis=0),
                label = "%s %d" % (StreamName,StreamIdx),**plt_args)
        if plt_stderr:
            ax.fill_between(np.linspace(-lpad,rpad,nsamples),
                            data_ar.mean(axis=0)-sem(data_ar,axis=0),
                            data_ar.mean(axis=0)+sem(data_ar,axis=0),
                            alpha = 0.4,**plt_args)
        if ylim:
            ax.set_ylim(ylim)

        f.suptitle("Event %s,N=%d" % (EventName,nAvg))
        f.savefig(os.path.join(pltdir,"Event_%s_Triggered_%s_Idx%d.png" % (EventName,StreamName,StreamIdx)),
                  dpi = 300,transparent=True)
        return f

    def EventTriggeredEMG(self, EventTimes, EventName,
                          DigaChan=1,MastChan=2,
                          lpad=0.2,rpad=0.2, 
                          time_buckets = None,
                          plt_stderr=True, ylim=False,
                          convolve_s = None,pltdir = '.',**kwargs):
        """"""
        if 'EMGpltargs' in kwargs.keys():
            EMG_plt_args = args['EMGpltargs']
        else:
            EMG_plt_args = {'digastric':{'color':'black'},
                            'maseter':{'color':'red'}}
        fs = self.tdt.streams.EMGx.fs
        # raw data option
        if convolve_s is None:        
            digRC = zscore(self.EMG[DigaChan,:])
            masRC = zscore(self.EMG[MastChan,:])
        else:
            if convolve_s == self._convolve_s:
                # cache of previous computation, sloppy
                digRC = self._digRC
                masRC = self._masRC
            else:
                digRC = zscore(np.convolve(np.abs(self.EMG[DigaChan,:]),
                               np.ones((int(convolve_s*fs),))/int(convolve_s*fs),mode = 'same'))
                masRC = zscore(np.convolve(np.abs(self.EMG[MastChan,:]),
                                 np.ones((int(convolve_s*fs),))/int(convolve_s*fs),mode = 'same'))
                self._digRC = digRC
                self._masRC = masRC
                self._convolve_s = convolve_s
        os.makedirs(pltdir, exist_ok=True)

        dp_lpad = self._ts_EMGx_idx(lpad)
        dp_rpad = self._ts_EMGx_idx(rpad)

        nAvg = len(EventTimes)
        f,ax = plt.subplots(1,1)

        nsamples = dp_lpad+dp_rpad

        # # Create a vector from 0 up to nsamples
        sample_idx = np.arange(nsamples)


        # # Calculate the index of the first sample for each chunk
        # # Require integers, because it will be used for indexing

        ## drop spike times that I will not be able to average around 
        ## (i.e. clipped by beginning and end of file)
        start_idx = ((EventTimes - lpad) * fs).astype(int)
        start_idx = start_idx[(start_idx+nsamples)<len(digRC)-1]

        # # Use broadcasting to create an array with indices
        # # Each row contains consecutive indices for each chunk
        idx = start_idx[:, None] + sample_idx[None, :]
        
        # # Get all the chunks using fancy indexing
        print(idx.shape)
        # hey
        dig_ar = digRC[idx]
        mas_ar = masRC[idx]

        ax.plot(np.linspace(-lpad,rpad,nsamples),dig_ar.mean(axis=0),label = 'digastric',**EMG_plt_args['digastric'])
        ax.plot(np.linspace(-lpad,rpad,nsamples),mas_ar.mean(axis=0),label = 'maseter',**EMG_plt_args['maseter'])
        if plt_stderr:
            ax.fill_between(np.linspace(-lpad,rpad,nsamples),
                            dig_ar.mean(axis=0)-sem(dig_ar,axis=0),
                            dig_ar.mean(axis=0)+sem(dig_ar,axis=0),
                            alpha = 0.4,
                            **EMG_plt_args['digastric'])
            ax.fill_between(np.linspace(-lpad,rpad,nsamples),
                            mas_ar.mean(axis=0)-sem(mas_ar,axis=0),
                            mas_ar.mean(axis=0)+sem(mas_ar,axis=0),
                            alpha = 0.4,
                            **EMG_plt_args['maseter'])
        if ylim:
            ax.set_ylim(ylim)

        f.suptitle("Event %s,N=%d" % (EventName,nAvg))
        f.savefig(os.path.join(pltdir,"Event_%s_TriggeredEMG.png" % (EventName)),
                  dpi = 300,transparent=True)
        return f

    def WaterFallEMG(self,times,lpad,rpad,chans = [1,2],ztrans=True,
                     sig_yoff = 30, trial_yoff = 100,plt_args=None):
        f,ax = plt.subplots(1,1)
        clr = ['black','blue','red','green']
        for i,time in enumerate(times):
            xs,data = self.EMGx(time-lpad,time+rpad,ztrans=ztrans)
            for chan in chans:
                ax.plot(xs-time, 
                        data[chan,:]+(sig_yoff*(chan-min(chans)))+(i*trial_yoff),
                        color = clr[chan],**plt_args)
        return (f,ax)

    def OscPanel(self,start,stop,wires,EMG_chns=None):
        _unitdf = self.unitdf.reset_index()
        wgb = _unitdf.groupby('wire')
        times = (start,stop)
        if EMG_chns is not None:
            f, axar = plt.subplots(len(wires)+len(EMG_chns),1,sharex='all')
            for ei,chn in enumerate(EMG_chns):
                xs,EMGdata = self.EMGx(start,stop)
                axar[(ei+1)*-1].plot(xs,EMGdata[chn,:],lw = 0.8, color='black')
        else:
            f, axar = plt.subplots(len(wires),1,sharey='all',sharex='all')
        cmap=plt.get_cmap('tab20')
        for i,wn in enumerate(wires):
            wg = wgb.get_group(wn)
            xs,pNeu =  self.pNeu(*times)
            axar[i].plot(xs, pNeu[wn-1,:],color='black',linewidth = 0.75)
            print(min(xs),max(xs))
            axar[i].set_xlim(min(xs),(max(xs)-min(xs))*1.25+min(xs))
            nm_units_here = wg[wg.TDTts.between(*times)]['NEXSC'].nunique()
            SC_cnt = 0
            for ii,(sc,g) in enumerate(wg.groupby('NEXSC')):
                if sc==0:
                    continue
                g_mask = g.TDTts.between(*times)
                if g_mask.sum()>0:
                    axar[i].eventplot(g[g_mask]['TDTts'].values, 
                                      lineoffsets=(0.3*(SC_cnt/nm_units_here))+0.65,
                                      linelengths=0.3/nm_units_here,
                                      transform = axar[i].get_xaxis_transform(),
                                      color = cmap(SC_cnt/nm_units_here))
                    segs = np.zeros(self.waveforms[(wn,sc)][g_mask,:].shape+(2,))
                    segs[:,:,0] = np.r_[0:30]
                    segs[:,:,1] = self.waveforms[(wn,sc)][g_mask,:]
                    axins = axar[i].inset_axes([0.82,(SC_cnt)/nm_units_here,0.18,1/nm_units_here])
                    f.add_axes(axins)
                    axins.patch.set_alpha(0.02)
                    snips = LineCollection(segs, linewidths=0.25,
                                           colors=cmap(SC_cnt/nm_units_here), 
                                           linestyle='solid')
                    axins.add_collection(snips)
                    ymin, ymax = np.min(segs[:,:,1]),np.max(segs[:,:,1])
                    axins.set_ylim(ymin,ymax)
                    axins.set_xlim(0,30)
                    axins.text(0,0,"Wr%d:SC%d" % (wn,sc),size=6,transform = axins.transAxes)
                    [x.set_visible(False) for x in [axins.xaxis, axins.yaxis]]
                    SC_cnt+=1
        return f,axar

    def find_optotagged(self, dt=0.001, wn=0.01, mask = None, laser_epoc = 'LsrP'):
        random_ts = self.tdt.epocs[laser_epoc].onset-0.5
        laser_onsets = self.tdt.epocs[laser_epoc].onset
        bins = np.arange(0,self.tdt.info.duration.total_seconds(),dt)
        # have to find times that avoiding the laser
        df_dicts = []
        for (w,sc),g in self.unitdf.groupby(self.unitdf.index):
            if sc==0:
                continue
            dst = descritized_spike_train(g['TDTts'].values,bins)
            spt_baseline = descritized_spike_raster(random_ts,dst,dt,int((wn/dt)*10))
            spt_test = descritized_spike_raster(laser_onsets,dst,dt,int((wn/dt)*10))
            #if ~(np.any(spt_baseline) | np.any(spt_test)):
            #    continue
            p,I = SALTY(spt_baseline,spt_test,dt=dt,wn=wn)
            df_dicts.append({'wire':w,'sc':sc,'P':p,'Idiff':I})
        tagDf = pd.DataFrame(df_dicts).set_index(['wire','sc'])
        tagDf['LsrSig'] = tagDf['P']<0.05
        return tagDf

    def DeMultiPlex(self, plexed_names = ['Valv','Spkr','CamS']):
        # just a pass through
        DeMultiPlex(self.tdt, plexed_names=plexed_names)

# should really be able to de MultiPlex a free standing TDT block
def DeMultiPlex(TDT_d, plexed_names = ['Valv','Spkr','CamS']):
    if 'MPlx' not in TDT_d.epocs.keys():
        print("No MPlx Store, Not doing anything")
        return
    MPlex = TDT_d.epocs.MPlx
    dint = (MPlex.data).astype(np.uint8)
    rlens, rstrt, values = rle(dint)
    print("longest run is %d" % np.max(rlens))
    r_idxs = np.where(rlens>1)
    rdelta = (np.diff(np.c_[MPlex.onset[rstrt[r_idxs]],MPlex.onset[rstrt[r_idxs]+1]])*1000).flatten()
    if np.size(rdelta)!=0:
        assert(np.max(rdelta)//(1000/24414.1)<2),"There are runs of same data that have more than one clock tick"
        # now filter by the run starts, to drop duplicate events entering the same data
    MPlex.onset, MPlex.offset, MPlex.data = MPlex.onset[rstrt], MPlex.offset[rstrt], MPlex.data[rstrt]
    dint = (MPlex.data).astype(np.uint8)
    # unpack, reshape, and flip the column order, 
    # so now will be row for each event,  first store, the second store etc.
    unpacked = np.unpackbits(dint).reshape((-1,len(np.unpackbits(dint))//len(dint)))[:,::-1]
    [p for p in zip(MPlex.onset,unpacked)]
    # now I can slice through the columns to construct onset/offset type stores
    # again use rle here
    for ii, name in enumerate(plexed_names):
        if np.sum(unpacked[:,ii])==0:
            print("No %s event?" % (name))
            continue
        _,_idxs,_ = rle(unpacked[:,ii])
        onset_mask = np.where(unpacked[_idxs,ii]==1)
        offset_mask = np.where(unpacked[_idxs,ii]==0)
        if np.size(onset_mask)==0:
            # shit
            print("there is no start for %s, just set to zero" % name)
            onsets = np.r_[0]
        else:
            onsets = MPlex.onset[_idxs[onset_mask]]
            offsets = MPlex.offset[_idxs[offset_mask]]
            # only keep offsets after the first onset
        offsets = offsets[offsets>onsets[0]]
        # then check to see if last offset exists
        print(len(onsets),len(offsets))
        data = np.ones(len(offsets),dtype=np.uint8)
        TDT_d.epocs[name] = tdt.StructType({'name':name,
                                       'onset':onsets,
                                       'offset':offsets,
                                       'type':MPlex['type'],
                                       'type_str':MPlex['type_str'],
                                       'data':data,
                                       'dform':MPlex['dform'],
                                       'size':MPlex['size']})
    # once I have de-multiplexed, pop this off the epocs keys
    # so its gone!
    TDT_d.epocs.__dict__.pop('MPlx')


@njit
def trig_rate(ev_times, spike_times,lshift,rshift):
    rates = np.zeros(len(ev_times),dtype = np.float64)
    for ii in range(len(ev_times)):
        rates[ii]=np.sum((spike_times>=(ev_times[ii]-lshift))&(spike_times<(ev_times[ii]+rshift)))/(lshift+rshift)
    return np.nanmean(rates), np.nanstd(rates), rates

@njit
def trig_vec(ev_times, spike_times,lshift,rshift,bin_width):
    bins = np.arange(0,lshift+rshift+bin_width*0.01,bin_width)
    xs = bins[0:-1]
    rate_vecs = np.zeros((len(ev_times),)+xs.shape,dtype = np.float64)
    for ii in range(len(ev_times)):
        _spikes = spike_times[(spike_times>=(ev_times[ii]-lshift))&(spike_times<(ev_times[ii]+rshift))]-ev_times[ii]+lshift
        counts,_ =np.histogram(_spikes,bins)
        rate_vecs[ii,:]=counts/bin_width
    return rate_vecs, xs

@njit
def descritized_spike_train(spike_times, bins):
    has_spk = np.digitize(spike_times,bins)
    dst = np.zeros(bins.shape,dtype=np.bool8)
    dst[has_spk] = True
    return dst
    
@njit
def descritized_spike_raster(event_times,dst,dt,Wn):
    '''event_times, times of each trigger (seconds)
    dst, discretized spike raster, with each bin dt in length
    dt, time step in seconds
    Wn, window width, number of time steps'''
    event_didxs = (event_times/dt).astype(np.int64)
    dsr = np.zeros((len(event_times),Wn),dtype=np.bool_)
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

@njit
def find_opto_artifact_idxs(spiketimes,lasertimes,window = 0.0008):
    # preindex array
    spikes_to_drop = np.repeat(np.array([-1],dtype=np.int64),10000)
    counter = 0
    for lasertime in lasertimes:
        artifact_spikes = np.where(np.abs(spiketimes - lasertime)<window)[0]
        spikes_to_drop[counter:counter+len(artifact_spikes)]=artifact_spikes
        counter+=len(artifact_spikes)
    return np.sort(spikes_to_drop[0:counter])

def BalloonProgram(tdt_d,start_idx,rate,fs,Op='PAOp',Dr='PADr', measures = False,
                   shoulder_pad=5, shoulder_dur=30):
    """rate in uL / sec"""
    inflt_start = tdt_d.epocs[Op].onset[start_idx]
    inflt_end = tdt_d.epocs[Op].offset[start_idx]
    deflt_start = tdt_d.epocs[Op].onset[start_idx+1]
    deflt_end = tdt_d.epocs[Op].offset[start_idx+1]
    inflt_dur = inflt_end-inflt_start
    inflt_vol = inflt_dur*rate
    deflt_dur = deflt_end-deflt_start
    vol = np.r_[np.linspace(0,inflt_vol,int(inflt_dur*fs)),
                np.repeat(inflt_vol,int((deflt_start-inflt_end)*fs)),
                np.linspace(inflt_vol,0,int(deflt_dur*fs))]
    xs = np.linspace(inflt_start,deflt_end,len(vol))
    # want to return corner times? for measurement extraction
    if measures:
        prs_xs = np.linspace(0,tdt_d.info.duration.total_seconds(),len(tdt_d.streams.Vprs.data))
        pre_m = (prs_xs>inflt_start-shoulder_pad-shoulder_dur)&(prs_xs<inflt_start-shoulder_pad)
        inflt_m = (prs_xs>inflt_end)&(prs_xs<deflt_start)
        post_m = (prs_xs>deflt_end+shoulder_pad)&(prs_xs<deflt_end+shoulder_pad+shoulder_dur)
        msrs = []
        for _m in [pre_m,inflt_m,post_m]:
            msrs.append(tdt_d.streams.Vprs.data[_m].mean())
        return (xs,vol,msrs)
    else:    
        return (xs,vol)

# pull laser stimulation blocks out from the headers
# without having to read the whole block
def get_lsr_stim_blocks(header,laser_name = 'LsrP',maxISI=2,minNPulse=5):
    import pandas as pd
    from datetime import timedelta, datetime
    # get the duration of the file in seconds from the header
    dur = (datetime.fromtimestamp(header.stop_time[0])-\
           datetime.fromtimestamp(header.start_time[0])).total_seconds()
    if laser_name not in header.stores.keys():
        return None
    # have to contstruct the stimulus epocs
    lsr_burst_start_idxs = np.where((np.diff(\
            np.r_[-maxISI*1.1,header.stores[laser_name].onset][::-1])[::-1])<-maxISI)[0]
    lsr_burst_end_idxs = np.where(np.diff(\
        np.r_[header.stores[laser_name].onset,dur+maxISI*1.1])>maxISI)[0]
    # I have some singletons in here, and some general garbage
    LsrStimIdxs = np.c_[lsr_burst_start_idxs,lsr_burst_end_idxs]
    # drop all the stimulus burst with fewer than 5 stimulus
    drop_idxs = np.where(np.diff(LsrStimIdxs).flatten()<minNPulse)[0]
    mask = np.ones(len(LsrStimIdxs), np.bool_)
    mask[drop_idxs] = 0
    LsrStimIdxs = LsrStimIdxs[mask]
    LsrStimuli = header.stores[laser_name].onset[LsrStimIdxs]
    LsrStimFreqs = np.diff(LsrStimIdxs).flatten()/np.diff(LsrStimuli).flatten()
    stim_durs = np.diff(LsrStimuli)
    # now make a data frame
    return pd.DataFrame({'BurstStartTime':LsrStimuli[:,0],
                  'BurstEndTime':LsrStimuli[:,1],
                  'BurstFreq':LsrStimFreqs})

# keep this here to simplify other stuff.
def make_bursts(sig,xs,prom=8.5,startISI=0.8,endISI=1,minN=3):
    from scipy.signal import find_peaks
    # find antipeaks
    apks,_ = find_peaks(sig*-1,prominence=prom)
    # find antipeaks
    pks,_ = find_peaks(sig,prominence=prom)
    all_peaks = np.sort(np.r_[xs[pks],xs[apks]])
    # lets just do the burst definitionally manually.
    # make a min ISI, to start a burst, make a max ISI, to end a burst,
    # make a min event number to keep burst
    burst_start_idxs = np.where((np.diff(np.r_[-startISI*1.1,all_peaks][::-1])[::-1])<-startISI)[0]
    burst_end_idxs = np.where(np.diff(np.r_[all_peaks,all_peaks[-1]+endISI*1.1])>endISI)[0]
    burst_start_idxs_u = np.setdiff1d(burst_start_idxs,burst_end_idxs)
    burst_end_idxs_u = np.setdiff1d(burst_end_idxs,burst_start_idxs)
    # only keep burst starts that are smaller than the largest burst end
    burst_start_idxs_u = burst_start_idxs_u[burst_start_idxs_u<burst_end_idxs_u[-1]]
    insert_starts = np.searchsorted(burst_end_idxs_u,burst_start_idxs_u)
    if np.any(np.diff(insert_starts)==0):
        # if there are ambiguous starts to the same end, use the earlier start
        fix_starts = burst_start_idxs_u[np.r_[np.diff(insert_starts).astype(np.bool_),True]]
    else:
        fix_starts = burst_start_idxs_u
    # now check the burst ends
    insert_ends = np.searchsorted(fix_starts,burst_end_idxs_u)
    if np.any(np.diff(insert_ends)==0):
        fix_ends = burst_end_idxs_u[np.r_[np.diff(insert_ends).astype(np.bool_),True]]
    else:
        fix_ends = burst_end_idxs_u
    bursts = all_peaks[np.c_[fix_starts,fix_ends]]
    burst_lens = np.diff(np.c_[fix_starts,fix_ends])
    # drop burst with fewer than 3 events
    bursts = bursts[(burst_lens>minN).flatten()]
    return bursts

def events_to_bursts(onset, rec_t_start, rec_t_end, maxISI, minNPulse):
        # have to contstruct the stimulus epocs
    lsr_burst_start_idxs = np.where((np.diff(\
            np.r_[rec_t_start-maxISI*1.1,onset][::-1])[::-1])<-maxISI)[0]
    lsr_burst_end_idxs = np.where(np.diff(\
        np.r_[onset,rec_t_end+maxISI*1.1])>maxISI)[0]
    # I have some singletons in here, and some general garbage
    LsrStimIdxs = np.c_[lsr_burst_start_idxs,lsr_burst_end_idxs]
    # drop all the stimulus burst with fewer than 5 stimulus
    drop_idxs = np.where(np.diff(LsrStimIdxs).flatten()<minNPulse)[0]
    mask = np.ones(len(LsrStimIdxs), np.bool_)
    mask[drop_idxs] = 0
    LsrStimIdxs = LsrStimIdxs[mask]
    LsrStimuli = onset[LsrStimIdxs]
    LsrStimFreqs = np.diff(LsrStimIdxs).flatten()/np.diff(LsrStimuli).flatten()
    return pd.DataFrame({'BurstStartTime':LsrStimuli[:,0],
              'BurstEndTime':LsrStimuli[:,1],
              'BurstFreq':LsrStimFreqs})
    
def tdt_ts_to_mov_ts(tdt_header,movie,sec):
    import warnings
    from TDTNex import get_avg_fps_float
    from scipy.stats import mode
    # make the seconds an array:
    if type(sec)!=type(np.array([])): 
        sec = np.array(sec)
    # flatten and sort the times
    sec = np.sort(sec.flatten())
    # figure the frame times:
    if ('FrmN' in tdt_header.stores.keys()) and\
       ('CamF' in tdt_header.stores.keys()):
        FN='CamF'
        warnings.warn("equivocal frame time epoc names, using %s" % FN)
    elif 'FrmN' in tdt_header.stores.keys():
        FN='FrmN'
    elif 'CamF' in tdt_header.stores.keys():
        FN='CamF'
    else:
        warnings.warn("no timing epoc name that I guess")
    # drop a time zero frame if it exists:
    if tdt_header.stores[FN].onset[0]==0:
        # drop the first frame onset
        tdt_header.stores[FN].onset=tdt_header.stores[FN].onset[1:]
        tdt_header.stores[FN].offset=tdt_header.stores[FN].offset[1:]
        tdt_header.stores[FN].data=tdt_header.stores[FN].data[1:]
    # check that there are no gaps in the frame sequence:
    td = np.diff(tdt_header.stores[FN].onset)
    assert(np.max(td)<(mode(td).mode*1.3)),\
    "max frame interval %.6f\
    is more than 1.3 times mode interval\
    %.6f" % (np.max(td),mode(td).mode)
    # if this is all good to here lets make the frame times 
    # a variable
    frame_xs = tdt_header.stores[FN].onset
    # finally compute the ts
    avg_fps = get_avg_fps_float(movie)
    frame_idxs = np.searchsorted(frame_xs,sec)
    # what to do if sec is before or after last time?
    if 0 in frame_idxs:
        warnings.warn("frame idx 0, maybe you have picked\n\
        a time before the first frame in the movie, %.6f" %\
                      (frame_xs[0]))
    if len(frame_xs) in frame_idxs:
        warnings.warn("last frame picked, maybe you have picked\n\
        a time after the first frame in the movie, %.6f" %\
                      (frame_xs[-1]))
    movie_times = frame_idxs/avg_fps
    # just do this as a list:
    movie_ts = []
    for _t in movie_times:
        # now to formate the movie seconds as a time stamp
        H=_t//3600
        M=(_t%3600)//60
        S=((_t%3600)%60)//1
        ms=(((_t%3600)%60)%1)*1000//1
        movie_ts.append("%02d:%02d:%02d.%03d" % (H,M,S,ms))
    return movie_ts

# helpers for animated movie generation
class VidDataStream(object):
    def __init__(self,Data,Fs,RngLw,RngHi,ReadOffsetShimDP=0):
        self.Data = Data
        print(len(self.Data))
        self.Fs = Fs
        self.RngLw = RngLw
        self.RngHi = RngHi
        self.Rng = self.RngHi - self.RngLw
        self.ReadOffsetShimDP=ReadOffsetShimDP

    def ds_for_vid(self,frame_width,time_span):
        sec_per_pixel = time_span/frame_width
        emg_pts_per_pixel = int(self.Fs*sec_per_pixel)
        y = self.Data[0:(len(self.Data)//emg_pts_per_pixel)*emg_pts_per_pixel]
        t_end_px = len(y)/self.Fs+self.ReadOffsetShimDP/self.Fs
        t_strt_px = self.ReadOffsetShimDP/self.Fs
        print(t_end_px)
        ds_min = np.min(y.reshape((-1,emg_pts_per_pixel)),axis=1)
        ds_max = np.max(y.reshape((-1,emg_pts_per_pixel)),axis=1)
        # this is a little fun with the ravel, means I have essentially halved the time span?
        stream_clip=np.clip(np.ravel(np.c_[ds_min,ds_max]),
                            self.RngLw,
                            self.RngHi)
        x = np.linspace(t_strt_px,t_end_px,len(stream_clip))
        self.VdsFs = len(stream_clip)/(t_end_px-t_strt_px)
        # clip 
        # offset 
        stream_clip-=self.RngLw
        # range scale
        stream_clip/=self.Rng # now should be zero to 1
        # invert
        stream_clip*=-1 # invert so Y can be subtracted, opencv coords are Y mirrored
        # the rest I will have to do in the video, 
        # need to know the mapping to video coords
        # device scale
        ## stream_clip*=(data_height/2)
        # device offset
        ## stream_clip+=(out_frame_height+((ii-1)*data_height/2))
        # stream_clip.astype(np.int32)
        # WILL HAVE TO ADD INTERPOLATION HERE TOO,
        # FOR STREAMS WITH FEWER DP / SEC THAN PX / SEC.
        return x,stream_clip,self.VdsFs
        
# find movie# Now lets make a movie:
def MakeHLDataClip(tdt_d,movie,movie_start_time,movie_end_time,
                   VidDataStreams, time_span = 7.5,
                   pl_thickness = 1,
                   output_name = "ForceMovie", unit = None, rec = None,
                   video_range = 0.3, out_frame_rate=None, bitrate=1000):
    '''
    Make a movie from a synchronously recorded stream with tdt, and basler camera
    time_span is in seconds and sets the data presentation by downsampling, 
    greater time span is more downsampling.
    '''
    from scipy.signal import decimate
    from scipy.stats import mode
    import cv2
    cap = cv2.VideoCapture(movie)
    # get their frame widths and heights for output cap
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    # now, add below the frame, by the 'video_range'
    data_height = int(((frame_height*video_range)//2)*2)
    print(data_height)
    out_frame_height = frame_height+data_height
    print(out_frame_height)
    
    # get the frame rate
    FrameOnsets = tdt_d.epocs.FrmN.onset
    # check to make sure I 
    frame_rate = 1/mode(np.diff(FrameOnsets)).mode
    print(frame_rate)
    # find the frame indexes corresponding to the movie start and stop times
    frame_idx_start = np.argmin(np.abs(tdt_d.epocs.FrmN.onset-movie_start_time))
    frame_idx_end = np.argmin(np.abs(tdt_d.epocs.FrmN.onset-movie_end_time))
    MST_FA = tdt_d.epocs.FrmN.onset[frame_idx_start] # Movie Start Time Frame Accurate
    MET_FA = tdt_d.epocs.FrmN.onset[frame_idx_end] # Movie End Time Frame Accurate

    # make sure the closest frame is close enough
    assert(np.abs(movie_start_time-MST_FA)<0.05),"movie start time %g is too far away from closest frame %g" % (movie_start_time,FrameOnsets[frame_idx_start])
    assert(np.abs(movie_end_time-MET_FA)<0.05),"movie end time %g is too far away from closest frame %g" % (movie_end_time,FrameOnsets[frame_idx_end])
    # seek to the starting frame:
    print("FPS:",cap.get(cv2.CAP_PROP_FPS))
    assert cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx_start)
    # file name stuff
    import os
    dir_name = os.path.dirname(output_name)
    f_name = os.path.basename(output_name).split(os.extsep)[0]+'.mp4'
    output_path = os.path.join(dir_name,f_name)
    assert(not os.path.exists(output_path)),'Path %s already exits, I dont want to overwrite' % output_name

    # set up the output for the capture with gstreamer, to control the bitrate directly, this is nice!
    # set up the out cap to sink into gst.
    # old pipeline
    # appsink2mp4 = "appsrc ! \
    # videoconvert ! nvh265enc bitrate=%d ! \
    # h265parse ! mp4mux ! \
    # filesink location=%s" % (bitrate, output_path)
    # new pipeline, specify pixel format to be powerpoint compatible.
    appsink2mp4 = "appsrc ! \
                   videoconvert ! video/x-raw, format=NV12 ! nvh264enc bitrate=%d ! \
                   h264parse ! mp4mux ! \
                   filesink location=%s" % (bitrate, output_path)
    if out_frame_rate is None:
        out_frame_rate = frame_rate
        vid_dur = (MET_FA - MST_FA)
    else:
        vid_dur = (MET_FA - MST_FA)*(frame_rate/out_frame_rate)
    print(appsink2mp4)
    out = cv2.VideoWriter(appsink2mp4, cv2.CAP_GSTREAMER, 0, out_frame_rate, 
                          (frame_width,out_frame_height),True)
    #print('here')
    stream_dp = frame_width
    stream_offset = stream_dp//2
    VidDatas = []
    # have to work out the scaling and offsets for the two streams
    for ii,VDS in enumerate(VidDataStreams):
        x,y,adj_fs = VDS.ds_for_vid(frame_width,time_span)
        # device scale
        y*=(data_height/len(VidDataStreams))
        # device offset
        y+=(out_frame_height+((ii-len(VidDataStreams)+1)*(data_height/len(VidDataStreams))))
        VidDatas.append(y.astype(np.int32))
    Cols = np.arange(frame_width)
    VidData_a = np.array(VidDatas)
    print(VidData_a.shape)
    lx1,ly1 = frame_width//2, 0
    lx2,ly2 = frame_width//2, data_height
    text_org = frame_width-100,100
    blank_out_frame = np.zeros((out_frame_height,frame_width,3),dtype='uint8')
    print("blank_out_frame shape %s, frame width %d, frame height %d" % (blank_out_frame.shape.__repr__(), 
                                                                         frame_width,
                                                                         frame_height))
    line_thickness = 1
    # what to do about laser?
    # lets do this for the laser:
    if 'LsrP' in tdt_d.epocs.keys():
        # lets make laser stimulation bursts:
        hdr = tdt.read_block(
            os.path.join(
                tdt_d.info.tankpath,tdt_d.info.blockname),
            headers=1)
        LsrDf = get_lsr_stim_blocks(hdr)
        if len(LsrDf)>0:
            LsrDf['BurstFreqInt'] = np.round(LsrDf['BurstFreq']).astype(np.int32)
            # okay have stimulation bursts, lets make an epoc
            from tdt import StructType
            NewEpoc = {}
            for k in tdt_d.epocs.LsrP.keys():
                if k in ['onset','offset','data']:
                    NewEpoc[k]=np.array([])
                else:
                    NewEpoc[k]=tdt_d.epocs.LsrP[k]
            NewEpoc = StructType(NewEpoc)
            NewEpoc.name = 'Stt1'
            NewEpoc.onset = LsrDf.BurstStartTime.values
            NewEpoc.offset = LsrDf.BurstEndTime.values
            NewEpoc.data = np.round(LsrDf.BurstFreq.values).astype(np.int32)
            tdt_d.epocs['LsFq'] = NewEpoc
            # cute way to do this:
            # subtract the FrmN from these bracket Indexs.
            # take only the sign of the different
            # compute the diff of the brackets across axis=1
            # 
            LsFqFrameIdxs = np.searchsorted(tdt_d.epocs.FrmN.onset,
                                            np.c_[tdt_d.epocs.LsFq.onset,
                                                  tdt_d.epocs.LsFq.offset])
            match_row = lambda FN: np.argwhere(
                (np.diff(np.sign(LsFqFrameIdxs-FN),axis=1)
                    .flatten()>1)).flatten()
        else:
            match_row = lambda FN: np.array([])
    else:
        match_row = lambda FN: np.array([])
    
    while cap.isOpened():
        grabbed, frame = cap.read()
        if grabbed:
            # CAP_PROP_POS_FRAMES is the index position of the NEXT frame,
            # so the current frame index is one less
            framen = int(cap.get(cv2.CAP_PROP_POS_FRAMES))-1
            t = tdt_d.epocs.FrmN.onset[framen]
            if (framen==frame_idx_end):
                break
            # zero out old frame
            blank_out_frame[0:frame_height,:,:]=frame
            blank_out_frame[frame_height:,:,:]=0
            # index in to the decimated data
            VidDataidx = int(FrameOnsets[framen]*adj_fs)-stream_offset
            for i in range(len(VidDataStreams)):
                points = np.column_stack((Cols,VidData_a[i,VidDataidx:VidDataidx+stream_dp])).astype(np.int32)
                points = points.reshape((-1,1,2))
                cv2.polylines(blank_out_frame,[points],False,(0,255,255),pl_thickness,
                              cv2.LINE_AA)
            # draw a verticle line:
            cv2.line(blank_out_frame[frame_height:,:,:], (lx1, ly1), (lx2, ly2), (150, 150, 0), thickness=line_thickness)
            # see if we have laser stimulaiton happening here:
            row_match = match_row(framen)
            if np.size(row_match)==1:
                freq = "%d Hz" % LsrDf.loc[row_match,'BurstFreqInt'].iloc[0]
                cv2.putText(blank_out_frame, freq, (50,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 4, cv2.LINE_AA)
            out.write(blank_out_frame)
        else:
            break
    cap.release()
    out.release()    
    
    if unit is not None:
        assert(rec is not None),"if listening to units, assume merged sorts from OFS"
        # now I want to add the spiking unit data as audio next to this.
        # if the outputrate for the video is manually set, this could pose a problem.
        Adur = float(MET_FA-MST_FA)
        SU_fs = tdt_d.snips.eNeu.fs
        SU_audio_wv = np.zeros(int(Adur*tdt_d.snips.eNeu.fs))

        nsamples = 30 # this is fixed at the TDT digitization
        sample_idx = np.arange(nsamples)

        w,sc = unit
        # mask the unit to the period of the video
        u_m = rec.unitdf.loc[(w,sc)].TDTts.between(MST_FA,MET_FA)
        u_times = rec.unitdf.loc[(w,sc),'TDTts'][u_m].values
        # have to offset the u_times to begin at video start
        u_times-=MST_FA
        u_waves = rec.waveforms[(w,sc)][u_m]

        # find the corresponding index into the wavedata 
        SU_audio_wv_idxs = (u_times*SU_fs).astype(int)

        # now do the fancy indexing, with numpy broadcasting
        idx = SU_audio_wv_idxs[:,None] + sample_idx[None,:]
        SU_audio_wv[idx] = u_waves.reshape(idx.shape)

        # interpolate the waves, based on the output_frame_rate
        xp = np.linspace(0,Adur,len(SU_audio_wv))
        x = np.linspace(0,Adur,int(len(SU_audio_wv)*2*(frame_rate/out_frame_rate)))
        SU_audio_wv_intrp = np.interp(x,xp,SU_audio_wv)

        # now I just have to write the wave and join it to the video.
        from scipy.io.wavfile import write as writewav
        SU_data_wave_name = output_path.split(os.extsep)[0]+"audio_w%02d_sc%d.wav" % (w,sc)

        # instead of using rubber band, try just adjust the integer sample frequency in the wav file to m
        # to mach the length of the video, considering the length of the wav data.
        # have to amplify the audio, and clip, and cast to float32 for the wave format
        scld_SU_audio = np.clip((SU_audio_wv*10**4),-1,1).astype(np.float32)
        writewav(SU_data_wave_name, int(len(scld_SU_audio)/float(vid_dur)), scld_SU_audio)

        merge_run = ['ffmpeg','-i', output_path,'-i', SU_data_wave_name, 
                     '-c:v', 'copy', '-c:a', 'aac', 
                     output_path.split(os.extsep)[0]+"sound_w%02d_sc%d.mp4" % (w,sc)]
        subprocess.run(merge_run)

def LaserDataClip(tdt_d,movie,movie_start_time,movie_end_time, time_span = 7.5,
                   pl_thickness = 1,
                   output_name = "ForceMovie", unit = None, rec = None,
                   video_range = 0.3, out_frame_rate=None, bitrate=1000):
    '''
    Annotate Laser Activation in Movie.
    '''
    from scipy.stats import mode
    import cv2
    cap = cv2.VideoCapture(movie)
    # get their frame widths and heights for output cap
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    # now, add below the frame, by the 'video_range'
    out_frame_height = frame_height
    print(out_frame_height)
    
    # get the frame rate
    FrameOnsets = tdt_d.epocs.FrmN.onset
    # check to make sure I 
    frame_rate = 1/mode(np.diff(FrameOnsets)).mode
    print(frame_rate)
    # find the frame indexes corresponding to the movie start and stop times
    frame_idx_start = np.argmin(np.abs(tdt_d.epocs.FrmN.onset-movie_start_time))
    frame_idx_end = np.argmin(np.abs(tdt_d.epocs.FrmN.onset-movie_end_time))
    MST_FA = tdt_d.epocs.FrmN.onset[frame_idx_start] # Movie Start Time Frame Accurate
    MET_FA = tdt_d.epocs.FrmN.onset[frame_idx_end] # Movie End Time Frame Accurate

    # make sure the closest frame is close enough
    assert(np.abs(movie_start_time-MST_FA)<0.05),"movie start time %g is too far away from closest frame %g" % (movie_start_time,FrameOnsets[frame_idx_start])
    assert(np.abs(movie_end_time-MET_FA)<0.05),"movie end time %g is too far away from closest frame %g" % (movie_end_time,FrameOnsets[frame_idx_end])
    # seek to the starting frame:
    print("FPS:",cap.get(cv2.CAP_PROP_FPS))
    assert cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx_start)
    # file name stuff
    import os
    dir_name = os.path.dirname(output_name)
    f_name = os.path.basename(output_name).split(os.extsep)[0]+'.mp4'
    output_path = os.path.join(dir_name,f_name)
    assert(not os.path.exists(output_path)),'Path %s already exits, I dont want to overwrite' % output_name

    # set up the output for the capture with gstreamer, to control the bitrate directly, this is nice!
    # set up the out cap to sink into gst.
    # old pipeline
    # appsink2mp4 = "appsrc ! \
    # videoconvert ! nvh265enc bitrate=%d ! \
    # h265parse ! mp4mux ! \
    # filesink location=%s" % (bitrate, output_path)
    # new pipeline, specify pixel format to be powerpoint compatible.
    appsink2mp4 = "appsrc ! \
                   videoconvert ! video/x-raw, format=NV12 ! nvh264enc bitrate=%d ! \
                   h264parse ! mp4mux ! \
                   filesink location=%s" % (bitrate, output_path)
    if out_frame_rate is None:
        out_frame_rate = frame_rate
        vid_dur = (MET_FA - MST_FA)
    else:
        vid_dur = (MET_FA - MST_FA)*(frame_rate/out_frame_rate)
    print(appsink2mp4)
    out = cv2.VideoWriter(appsink2mp4, cv2.CAP_GSTREAMER, 0, out_frame_rate, 
                          (frame_width,out_frame_height),True)
    # what to do about laser?
    # lets do this for the laser:
    blank_out_frame = np.zeros((out_frame_height,frame_width,3),dtype='uint8')
    if 'LsrP' in tdt_d.epocs.keys():
        # lets make laser stimulation bursts:
        hdr = tdt.read_block(
            os.path.join(
                tdt_d.info.tankpath,tdt_d.info.blockname),
            headers=1)
        LsrDf = get_lsr_stim_blocks(hdr)
        if len(LsrDf)>0:
            LsrDf['BurstFreqInt'] = np.round(LsrDf['BurstFreq']).astype(np.int32)
            # okay have stimulation bursts, lets make an epoc
            from tdt import StructType
            NewEpoc = {}
            for k in tdt_d.epocs.LsrP.keys():
                if k in ['onset','offset','data']:
                    NewEpoc[k]=np.array([])
                else:
                    NewEpoc[k]=tdt_d.epocs.LsrP[k]
            NewEpoc = StructType(NewEpoc)
            NewEpoc.name = 'Stt1'
            NewEpoc.onset = LsrDf.BurstStartTime.values
            NewEpoc.offset = LsrDf.BurstEndTime.values
            NewEpoc.data = np.round(LsrDf.BurstFreq.values).astype(np.int32)
            tdt_d.epocs['LsFq'] = NewEpoc
            # cute way to do this:
            # subtract the FrmN from these bracket Indexs.
            # take only the sign of the different
            # compute the diff of the brackets across axis=1
            # 
            LsFqFrameIdxs = np.searchsorted(tdt_d.epocs.FrmN.onset,
                                            np.c_[tdt_d.epocs.LsFq.onset,
                                                  tdt_d.epocs.LsFq.offset])
            match_row = lambda FN: np.argwhere(
                (np.diff(np.sign(LsFqFrameIdxs-FN),axis=1)
                    .flatten()>1)).flatten()
        else:
            match_row = lambda FN: np.array([])
    else:
        match_row = lambda FN: np.array([])
    
    while cap.isOpened():
        grabbed, frame = cap.read()
        if grabbed:
            # CAP_PROP_POS_FRAMES is the index position of the NEXT frame,
            # so the current frame index is one less
            framen = int(cap.get(cv2.CAP_PROP_POS_FRAMES))-1
            t = tdt_d.epocs.FrmN.onset[framen]
            if (framen==frame_idx_end):
                break
            # zero out old frame
            blank_out_frame[0:frame_height,:,:]=frame
            blank_out_frame[frame_height:,:,:]=0
            # see if we have laser stimulaiton happening here:
            row_match = match_row(framen)
            if np.size(row_match)==1:
                freq = "%d Hz" % LsrDf.loc[row_match,'BurstFreqInt'].iloc[0]
                cv2.putText(blank_out_frame, freq, (50,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 4, cv2.LINE_AA)
            out.write(blank_out_frame)
        else:
            break
    cap.release()
    out.release()    
    
    if unit is not None:
        assert(rec is not None),"if listening to units, assume merged sorts from OFS"
        # now I want to add the spiking unit data as audio next to this.
        # if the outputrate for the video is manually set, this could pose a problem.
        Adur = float(MET_FA-MST_FA)
        SU_fs = tdt_d.snips.eNeu.fs
        SU_audio_wv = np.zeros(int(Adur*tdt_d.snips.eNeu.fs))

        nsamples = 30 # this is fixed at the TDT digitization
        sample_idx = np.arange(nsamples)

        w,sc = unit
        # mask the unit to the period of the video
        u_m = rec.unitdf.loc[(w,sc)].TDTts.between(MST_FA,MET_FA)
        u_times = rec.unitdf.loc[(w,sc),'TDTts'][u_m].values
        # have to offset the u_times to begin at video start
        u_times-=MST_FA
        u_waves = rec.waveforms[(w,sc)][u_m]

        # find the corresponding index into the wavedata 
        SU_audio_wv_idxs = (u_times*SU_fs).astype(int)

        # now do the fancy indexing, with numpy broadcasting
        idx = SU_audio_wv_idxs[:,None] + sample_idx[None,:]
        SU_audio_wv[idx] = u_waves.reshape(idx.shape)

        # interpolate the waves, based on the output_frame_rate
        xp = np.linspace(0,Adur,len(SU_audio_wv))
        x = np.linspace(0,Adur,int(len(SU_audio_wv)*2*(frame_rate/out_frame_rate)))
        SU_audio_wv_intrp = np.interp(x,xp,SU_audio_wv)

        # now I just have to write the wave and join it to the video.
        from scipy.io.wavfile import write as writewav
        SU_data_wave_name = output_path.split(os.extsep)[0]+"audio_w%02d_sc%d.wav" % (w,sc)

        # instead of using rubber band, try just adjust the integer sample frequency in the wav file to m
        # to mach the length of the video, considering the length of the wav data.
        # have to amplify the audio, and clip, and cast to float32 for the wave format
        scld_SU_audio = np.clip((SU_audio_wv*10**4),-1,1).astype(np.float32)
        writewav(SU_data_wave_name, int(len(scld_SU_audio)/float(vid_dur)), scld_SU_audio)

        merge_run = ['ffmpeg','-i', output_path,'-i', SU_data_wave_name, 
                     '-c:v', 'copy', '-c:a', 'aac', 
                     output_path.split(os.extsep)[0]+"sound_w%02d_sc%d.mp4" % (w,sc)]
        subprocess.run(merge_run)
        
def addSB(ax, sb_anch, xw, yh,
          x_lab_fmt = None, y_lab_fmt = None,
          sblw = 1.5, text_size = 6,
          bar_xoff=0,
          bar_yoff=2,
          ylab_yoff=1,ylab_xoff=1,
          xlab_yoff=1,xlab_xoff=1,
          bar_color = 'black'): 
    from matplotlib.lines import Line2D
    from matplotlib import transforms
    ax.get_xlim()
    ax.get_ylim()
    f = ax.get_figure()
    
    # use a scaled transulation to introduce an offset between the x and y scale bars.
    dx, dy = bar_xoff/72., (sblw*bar_yoff)/72.
    bar_offset = transforms.ScaledTranslation(dx, dy, f.dpi_scale_trans)
    bar_pad_transform = ax.transData + bar_offset
    
    # ylab offset
    dx, dy = (sblw*ylab_xoff)/72., (sblw*ylab_yoff)/72.
    ylab_offset = transforms.ScaledTranslation(dx, dy, f.dpi_scale_trans)
    ylab_pad_transform = ax.transData + ylab_offset
    
    # xlab offset
    dx, dy = (sblw*xlab_xoff)/72., (sblw*xlab_yoff)/72.
    xlab_offset = transforms.ScaledTranslation(dx, dy, f.dpi_scale_trans)
    xlab_pad_transform = ax.transData - xlab_offset
    
    # calculate the position of scale bars
    sb_anch_Disp = ax.transAxes.transform(sb_anch)
    data_inv = ax.transData.inverted()
    x,y = ax.transData.inverted().transform(sb_anch_Disp)
    
    xsb = Line2D(np.r_[x,x+xw],np.r_[y,y],lw = sblw, color = bar_color,
                 zorder = 12, clip_on = False)
    if x_lab_fmt is not None:
        ax.text(x+xw/2,y, x_lab_fmt % xw,
                transform = xlab_pad_transform, va = 'top', ha = 'center', 
                size = text_size, clip_on = False)
    ysb = Line2D(np.r_[x,x],np.r_[y,y+yh],lw = sblw, color = bar_color, 
                 transform = bar_pad_transform,zorder = 12, clip_on = False)
    if y_lab_fmt is not None:
        ax.text(x,y+yh/2, y_lab_fmt % yh,transform = ylab_pad_transform,va = 'center',
                size = text_size, clip_on = False)
    ax.add_artist(xsb)
    ax.add_artist(ysb)        

def drop_bad_first_FrmN(tdt_d,epoc_name = 'FrmN'):
    assert(epoc_name in tdt_d.epocs.keys()),"Frame Name %s not in epocs names" % (epoc_name)
    if tdt_d.epocs[epoc_name].onset[0]==0:
        print("Dropping first %s with onset time %g" % (epoc_name, tdt_d.epocs[epoc_name].onset[0]))
        tdt_d.epocs[epoc_name].onset = tdt_d.epocs[epoc_name].onset[1:]
        tdt_d.epocs[epoc_name].offset = tdt_d.epocs[epoc_name].offset[1:]
        tdt_d.epocs[epoc_name].data = tdt_d.epocs[epoc_name].data[1:]
    

# World collide, lets get helpers to work with Open Ephys data here as well
def MakeOEDataClip(OE_FrameOnsets,first_OE_timestamp,movie,movie_start_time,movie_end_time,
                   VidDataStreams, StreamReadRelSecOffset=0,
                   time_span = 7.5,
                   pl_thickness = 1,
                   LsrDf=None,
                   output_name = "ForceMovie",
                   unit=None,
                   video_range = 0.3, out_frame_rate=None, bitrate=1000):
    '''
    Make a movie from a synchronously recorded stream for OE, and manta? camera
    time_span is in seconds and sets the data presentation by downsampling, 
    greater time span is more downsampling.
    '''
    from scipy.signal import decimate
    from scipy.stats import mode
    import cv2
    cap = cv2.VideoCapture(movie)
    # get their frame widths and heights for output cap
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    # now, add below the frame, by the 'video_range'
    data_height = int(((frame_height*video_range)//2)*2)
    out_frame_height = frame_height+data_height
    
    # get the frame rate
    frame_rate = 1/mode(np.diff(OE_FrameOnsets)).mode
    # find the frame indexes corresponding to the movie start and stop times
    frame_idx_start = np.argmin(np.abs(OE_FrameOnsets-movie_start_time))
    frame_idx_end = np.argmin(np.abs(OE_FrameOnsets-movie_end_time))
    MST_FA = OE_FrameOnsets[frame_idx_start] # Movie Start Time Frame Accurate
    MET_FA = OE_FrameOnsets[frame_idx_end] # Movie End Time Frame Accurate

    # make sure the closest frame is close enough
    assert(np.abs(movie_start_time-MST_FA)<0.05),"movie start time %g is too far away from closest frame %g" % (movie_start_time,OE_FrameOnsets[frame_idx_start])
    assert(np.abs(movie_end_time-MET_FA)<0.05),"movie end time %g is too far away from closest frame %g" % (movie_end_time,OE_FrameOnsets[frame_idx_end])
    # seek to the starting frame:
    print("FPS:",cap.get(cv2.CAP_PROP_FPS))
    assert cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx_start)
    # file name stuff
    import os
    dir_name = os.path.dirname(output_name)
    f_name = os.path.basename(output_name).split(os.extsep)[0]+'.mp4'
    output_path = os.path.join(dir_name,f_name)
    #assert(not os.path.exists(output_path)),'Path %s already exits, I dont want to overwrite' % output_name

    # set up the output for the capture with gstreamer, to control the bitrate directly, this is nice!
    # set up the out cap to sink into gst.
    # old pipeline
    # appsink2mp4 = "appsrc ! \
    # videoconvert ! nvh265enc bitrate=%d ! \
    # h265parse ! mp4mux ! \
    # filesink location=%s" % (bitrate, output_path)
    # new pipeline, specify pixel format to be powerpoint compatible.
    appsink2mp4 = "appsrc ! \
                   videoconvert ! video/x-raw, format=NV12 ! nvh264enc bitrate=%d ! \
                   h264parse ! mp4mux ! \
                   filesink location=%s" % (bitrate, output_path)
    if out_frame_rate is None:
        out_frame_rate = frame_rate
        vid_dur = (MET_FA - MST_FA)
    else:
        vid_dur = (MET_FA - MST_FA)*(frame_rate/out_frame_rate)
    print(appsink2mp4)
    out = cv2.VideoWriter(appsink2mp4, cv2.CAP_GSTREAMER, 0, out_frame_rate, 
                          (frame_width,out_frame_height),True)
    stream_dp = frame_width
    stream_offset = stream_dp//2
    VidDatas = []
    # have to work out the scaling and offsets for the two streams
    for ii,VDS in enumerate(VidDataStreams):
        x,y,adj_fs = VDS.ds_for_vid(frame_width,time_span)
        print(adj_fs,ii)
        # device scale
        y*=(data_height/len(VidDataStreams))
        # device offset
        y+=(out_frame_height+((ii-len(VidDataStreams)+1)*(data_height/len(VidDataStreams))))
        VidDatas.append(y.astype(np.int32))
    Cols = np.arange(frame_width)
    VidData_a = np.array(VidDatas)
    print(VidData_a.shape,VidData_a.shape[1]/adj_fs)
    lx1,ly1 = frame_width//2, 0
    lx2,ly2 = frame_width//2, data_height
    text_org = frame_width-100,100
    blank_out_frame = np.zeros((out_frame_height,frame_width,3),dtype='uint8')
    print("blank_out_frame shape %s, frame width %d, frame height %d" % (blank_out_frame.shape.__repr__(), 
                                                                         frame_width,
                                                                         frame_height))
    line_thickness = 1
    # what to do about laser?
    # lets do this for the laser:
    if LsrDf is not None:
        LsrDf['BurstFreqInt'] = np.round(LsrDf['BurstFreq']).astype(np.int32)
        # okay have stimulation bursts, lets make an epoc
        LsFqOnset = LsrDf.BurstStartTime.values
        LsFqOffset = LsrDf.BurstEndTime.values
        LsFqIBFrq = np.round(LsrDf.BurstFreq.values).astype(np.int32)
        LsFqFrameIdxs = np.searchsorted(OE_FrameOnsets,
                                        np.c_[LsFqOnset,
                                              LsFqOffset])
        match_row = lambda FN: np.argwhere(
            (np.diff(np.sign(LsFqFrameIdxs-FN),axis=1)
                .flatten()>1)).flatten()
    else:
        match_row = lambda FN: np.array([])
    
    while cap.isOpened():
        grabbed, frame = cap.read()
        if grabbed:
            # CAP_PROP_POS_FRAMES is the index position of the NEXT frame,
            # so the current frame index is one less
            framen = int(cap.get(cv2.CAP_PROP_POS_FRAMES))-1
            t = OE_FrameOnsets[framen]
            if (framen==frame_idx_end):
                break
            # zero out old frame
            blank_out_frame[0:frame_height,:,:]=frame
            blank_out_frame[frame_height:,:,:]=0
            # index in to the decimated data
            # this is insane
            VidDataidx = int((OE_FrameOnsets[framen]-first_OE_timestamp-StreamReadRelSecOffset-time_span/4)*adj_fs)
            for i in range(len(VidDataStreams)):
                points = np.column_stack((Cols,VidData_a[i,VidDataidx:VidDataidx+stream_dp])).astype(np.int32)
                points = points.reshape((-1,1,2))
                cv2.polylines(blank_out_frame,[points],False,(0,255,255),pl_thickness,
                              cv2.LINE_AA)
            # draw a verticle line:
            cv2.line(blank_out_frame[frame_height:,:,:], (lx1, ly1), (lx2, ly2), (150, 150, 0), thickness=line_thickness)
            # see if we have laser stimulaiton happening here:
            row_match = match_row(framen)
            if np.size(row_match)==1:
                freq = "%d Hz" % LsrDf.loc[row_match,'BurstFreqInt'].iloc[0]
                cv2.putText(blank_out_frame, freq, (50,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 4, cv2.LINE_AA)
            out.write(blank_out_frame)
        else:
            break
    cap.release()
    out.release()    
    
    if unit is not None:
        pass
        # no finished with this yet, have to get snips for the phy2 kilosort output.


def sc_rate_intrp(time_series,xp):
    inst_rates = 1/np.diff(time_series)
    __xs = np.c_[time_series[0:-1],
                 time_series[1:]].flatten()
    __ys = np.repeat(inst_rates,2)
    return np.interp(xp,__xs,__ys)

@nb.njit
def mean_bucket(SS_idx_a,trace):
    bucket_mean = np.zeros(SS_idx_a.shape[0],dtype=np.float64)
    for i,(S,E) in enumerate(SS_idx_a):
        bucket_mean[i] = np.mean(trace[S:E])
    return bucket_mean
