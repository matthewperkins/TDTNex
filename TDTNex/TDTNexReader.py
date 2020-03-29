import neo
import quantities as pq
#%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.transforms as transforms
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import mode
import os
from pandas import IndexSlice as pidx

# maybe make a dataframe, row for each spike, 
## column indexs are wire sort code.
## values are EMGidx, pNueidx, nextime, tdt_time, and a waveform array?
# read the data in wire order frome the tdt, for each time, calculate the nex time
# grouby by wire, iterate through the spiketrains for each wire, use timestamp as a key to assign the sort code
# wire 
# maybe make a dataframe, row for each spike, 
## column indexs are wire sort code.
## values are EMGidx, pNueidx, nextime, tdt_time, and a waveform array?
# read the data in wire order frome the tdt, for each time, calculate the nex time
# grouby by wire, iterate through the spiketrains for each wire, use timestamp as a key to assign the sort code
# wire 
class TDTNex(object):
    def __init__(self, tdt_file_path, nex_file_path):
        """For the alignment and manipulation of TDT data files with manually cluster cutted data from Offline sorter,
        that has been exported to the Neurodata Explorer format. Note should submit pull request to change waveform offset
        in NEO to fix waveform bug (is 2 should be 4). Not sure how widely this applies. ALSO, DO NOT INVALIDATE waveforms 
        when cluster cutting, this will fuck up aligning the records back to TDT. Finally, add a neareset CameraFrames column
        to the coordiinated unit dataframe. If the nearest frame is > 11 milliseconds away, is NA. Also, alignment of the files
        works by exact matching of time stamps, which is brittle, because there is some weird clock skew in the NEX file
        that comes out of OffLineSorter. This skew is measured, and the expected neotime stamps are created by multipyling
        the TDT time stamps by the a coefficent. These expected time stamps are then rounded to 7 digits, and their
        matching timestamp is looked for on the appropriate wire from the NEX signals. Base on my experience the precision 
        of the rounding changes a bit from file to file, and may need to be tweaked"""
        self._tdt_fp = tdt_file_path
        self._nex_fp = nex_file_path
        self.tdt = tdt.read_block(self._tdt_fp)
        self.nex = NeuroExplorerIO(self._nex_fp)
        self.EMG = self.tdt.streams.EMGx.data
        self.seg = self.nex.read_segment()
        self._time_stamp_precision = 7 # this is critical, may need to adjust if readers are not aligning well.
        self.calc_sCoef() # side-effect function
        self._make_event_df() # side effect to add df to self.
        self._make_NexSort_df() # side-effect function add df to self
        self._make_Unit_df() # side-effect function add df to self
        
    def _ts_pNeu_idx(self,ts):
        return(int(self.tdt.streams.pNeu.fs*ts))
    
    def _ts_EMGx_idx(self,ts):
        return(int(self.tdt.streams.EMGx.fs*ts))
    
    def pNeu(self,start=None,stop=None):
        tdt_dur = self.tdt.info.duration.total_seconds()
        if start is not None:
            if (start>0)&(start<=tdt_dur):
                S = start
            else:
                print('Start arg is bad, setting to 0')
                S = 0
        else:
            S = 0
        if stop is not None:
            if (stop>0)&(stop>start)&(stop<tdt_dur):
                E = stop
            else:
                print('Stop arg is bad, setting to end of file')
        else:
            E = tdt_dur
        Sidx,Eidx = self._ts_pNeu_idx(S),self._ts_pNeu_idx(E)
        data = self.tdt.streams.pNeu.data[:,Sidx:Eidx]
        xs = np.linspace(S,E,data.shape[1])
        return xs,data

    def EMGx(self,start=None,stop=None):
        tdt_dur = self.tdt.info.duration.total_seconds()
        if start is not None:
            if (start>0)&(start<=tdt_dur):
                S = start
            else:
                print('Start arg is bad, setting to 0')
                S = 0
        else:
            S = 0
        if stop is not None:
            if (stop>0)&(stop>start)&(stop<tdt_dur):
                E = stop
            else:
                print('Stop arg is bad, setting to end of file')
        else:
            E = tdt_dur
        Sidx,Eidx = self._ts_EMGx_idx(S),self._ts_EMGx_idx(E)
        data = self.tdt.streams.EMGx.data[:,Sidx:Eidx]
        xs = np.linspace(S,E,data.shape[1])
        return xs,data

    def calc_sCoef(self):
        shared_names = np.intersect1d([k for k in self.tdt.epocs.keys()],
                                      [ev.name for ev in self.seg.events])
        ev_dict = {ev.name:ev for idx,ev in enumerate(self.seg.events)}
        if len(shared_names)==0:
            raise ValueError('No Shared events with which to calculate time skew')
        # find the first shared name with events:
        for shared_name in shared_names:
            if ev_dict[shared_name].times.size!=0:
                print("shared name: '%s'" % shared_name)
                break
        _nex_time = ev_dict[shared_name].times[-1].magnitude
        _tdt_time = self.tdt.epocs[shared_name].onset[-1]
        second_creep = (_tdt_time-_nex_time)/_tdt_time
        # store the sCoef, no return
        self.sCoef = 1-second_creep
        
    def _make_event_df(self):
        """Specific to epocs with offsets"""
        shared_names = np.intersect1d(sorted(self.tdt.epocs.keys()),
                                      sorted([ev.name for ev in self.seg.events]))
        ev_dict = {ev.name:ev for idx,ev in enumerate(self.seg.events)}
        #calc the length of event df
        len_ev_df = np.array([len(v.onset) for k,v in self.tdt.epocs.items()]).sum().astype('int')
        ev_df = pd.DataFrame({'name':['NA']*len_ev_df,
                              'onset':np.zeros((len_ev_df,),dtype=np.float),
                              'offset':np.zeros((len_ev_df,),dtype=np.float),
                              'data':np.zeros((len_ev_df,),dtype=np.float),
                              'onset_neo':np.zeros((len_ev_df,),dtype=np.float),
                              'offset_neo':np.zeros((len_ev_df,),dtype=np.float),
                              })
        _idx = 0
        for name in shared_names:
            tdt_ev = self.tdt.epocs[name]
            nex_ev_onset = ev_dict[name]
            nex_ev_offset = ev_dict[name[0:3]+'\\']
            assert len(nex_ev_onset==len(tdt_ev.onset)), "events of unequal length"
            ev_df.loc[_idx:_idx+len(tdt_ev.onset),'name'] = name
            ev_df.loc[_idx:_idx+len(tdt_ev.onset),'onset'] = tdt_ev.onset
            ev_df.loc[_idx:_idx+len(tdt_ev.onset),'offset'] = tdt_ev.offset
            ev_df.loc[_idx:_idx+len(tdt_ev.onset),'data'] = tdt_ev.data
            ev_df.loc[_idx:_idx+len(tdt_ev.onset),'onset_neo'] = nex_ev_onset.times.magnitude
            ev_df.loc[_idx:_idx+len(tdt_ev.onset),'offset_neo'] = nex_ev_offset.times.magnitude
            _idx+=len(tdt_ev.onset)    
        self.ev_df = ev_df

    def _make_Unit_df(self):
        from pandas import IndexSlice as pidx
        tdt = self.tdt
        frlen = len(tdt.snips.eNeu.ts)
        unitdf = pd.DataFrame({'wire':np.zeros((frlen,),dtype=np.int),
                              'SC':np.zeros((frlen,),dtype=np.int), # use -1 for unsorted
                              'TDTts':np.zeros((frlen,),dtype=np.float),
                              'NEOts':np.zeros((frlen,),dtype=np.float),
                              'EMGidx':np.zeros((frlen,),dtype=np.int),
                              'pNeoidx':np.zeros((frlen,),dtype=np.int)})
        # fill in the TDTts and NEOts by wire
        _idx_offset = 0
        pNeufs = tdt.streams.pNeu.fs
        EMGfs = tdt.streams.EMGx.fs
        for wire in np.r_[1:17]:
            _wt = tdt.snips.eNeu.ts[np.argwhere(tdt.snips.eNeu.chan.flatten()==wire).flatten()].flatten()
            unitdf.loc[_idx_offset:_idx_offset+len(_wt)-1,'wire']=wire
            unitdf.loc[_idx_offset:_idx_offset+len(_wt)-1,'TDTts']=_wt
            unitdf.loc[_idx_offset:_idx_offset+len(_wt)-1,'EMGidx']=(_wt*EMGfs).astype(int)
            unitdf.loc[_idx_offset:_idx_offset+len(_wt)-1,'pNeoidx']=(_wt*pNeufs).astype(int)
            unitdf.loc[_idx_offset:_idx_offset+len(_wt)-1,'NEOts']=np.round(_wt*self.sCoef,
                                                                           self._time_stamp_precision)
            #
            _idx_offset+=len(_wt)
        # now the painful loop, iterate over the wires, for each wire, 
        ## iterate over the sort codes, match times and assign the SC and waveforms
        unitdf.set_index(['wire','NEOts'],inplace=True)
        # update the sorts, and fill the waveforms
        nxdf = self.nex_df.reset_index().copy()
        waveforms = {}
        for (wn,SC,st_num),g in nxdf.groupby(['wire','SC','st_num']):
            # need to pull out the waves here.
            unitdf.loc[pidx[wn,g.st],'SC'] = SC
            # make sure that the index matching on time stamps has worked, this could be file and rounding dependent, is brittle.
            assert len(g.st)==len(unitdf.loc[pidx[wn,g.st],'SC']), "Not all time stamps are matched, only %d NexStamps are matched out of %d Total NexStamps"  % (len(g.st),len(unitdf.loc[pidx[wn,g.st],'SC']))

            # now get the waveforms
            _wvs = self.nex._get_spike_raw_waveforms(0,0,st_num,0*pq.s,
                                                     self.seg.t_stop)[:,0,:]

            # maybe the best thing to do is pack all the waveforms into list of arrays:
            # [wire,sortcode,np.array[spikenumber,[mV0:30]]]
            # in the data frame the wire and sort code are indexes, so just need to file in the spiketrainindex to point to right row in array
            waveforms[(wn,SC)]=np.copy(_wvs)
        self.unitdf = unitdf.reset_index().set_index(['wire','SC']).sort_index().copy()
        self.waveforms = waveforms

    def _make_NexSort_df(self):
        tdt = self.tdt
        # do the sort codes as integers, make unsorted = 0
        from string import ascii_lowercase
        SCdict = {ltr:SC+1 for SC,ltr in enumerate([x for x in ascii_lowercase[0:10]])}
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
        NexSorted_df = pd.DataFrame({'wire':np.zeros((nNexSpikes,),dtype=np.int),
                                     'SC':np.zeros((nNexSpikes,),dtype=np.int,),
                                     'st_num':np.zeros((nNexSpikes,),dtype=np.int,),
                                     'st':np.zeros((nNexSpikes,),dtype=np.float,)})
        _idx = 0
        for st_num, st in real_spktrns:
            wire = int(st.name[3:5])
            NexSorted_df.loc[_idx:_idx+len(st)-1,'wire']=wire
            SC = SCdict[st.name[5]]
            NexSorted_df.loc[_idx:_idx+len(st)-1,'SC']=SC
            NexSorted_df.loc[_idx:_idx+len(st)-1,'st_num']=st_num
            NexSorted_df.loc[_idx:_idx+len(st)-1,'st']=np.round(st.times.magnitude,
                                                                self._time_stamp_precision)
            _idx+=len(st)
        NexSorted_df.set_index(['wire','SC'],inplace=True)
        self.nex_df = NexSorted_df
        
    def UnitRaster(self,wire,sc,times,lpad,rpad):
        g = self.unitdf.groupby(['wire','SC']).get_group((wire,sc))
        try:
            iter(times)
        except TypeError:
            times=[times]
        nsnips = int(np.array([g.TDTts.between(t-lpad,t+rpad).sum() for t in times]).sum())
        evntsArray = np.zeros((nsnips,))
        evnts = []
        _seg_idx=0
        for t in times:
            _mask = g.TDTts.between(t-lpad,t+rpad)
            evnts.append(g[_mask]['TDTts'].values-t) # subtract t shift to zero
            evntsArray[_seg_idx:_seg_idx+_mask.sum()]=evnts[-1]
            _seg_idx+=_mask.sum()
        return(evnts,evntsArray)
            
    def AllUnitRasters(self,times,lpad,rpad,hist=True,fndec=None):
        # use TDT time, all in seconds
        plt_dir = os.path.join(os.curdir, "Rasters")
        os.makedirs(plt_dir,exist_ok=True)
        for (wire, sc),g  in self.unitdf.groupby(['wire','SC']):
            if sc==0:
                continue
            nsnips = int(np.array([g.TDTts.between(t-lpad,t+rpad).sum() for t in times]).sum())
            if nsnips<5:
                continue
            f= plt.figure()
            raster_ax = plt.axes([0.08,0.08,0.85,0.6])
            hist_ax = plt.axes([0.08,0.65,0.85,0.3])
            wf_ax = plt.axes([0.65,0.6,0.3,0.3])
            evnts = []
            evntsArray = np.zeros((nsnips,))
            # find out how many snips will be in the raster plot:
            # preindex a segs array for the line collection
            raster_segs = np.zeros((nsnips,30,2))
            # also pull out random segments for comparison, get 50 random if there are more than 50, 
            # otherwise grab them all.
            totsnips = len(g.TDTts)
            if totsnips>50:
                random_segs = np.zeros((50,30,2))
                random_segs[:,:,1] = self.waveforms[(wire,sc)][np.random.randint(0,totsnips-1,50)]
            else:
                random_segs = np.zeros((totsnips,30,2))
                random_segs[:,:,1] = self.waveforms[(wire,sc)][:]
            _seg_idx = 0
            for t in times:
                _mask = g.TDTts.between(t-lpad,t+rpad)
                evnts.append(g[_mask]['TDTts'].values-t) # subtract t shift to zero
                evntsArray[_seg_idx:_seg_idx+_mask.sum()]=evnts[-1]
                raster_segs[_seg_idx:_seg_idx+_mask.sum(),:,1]=self.waveforms[(wire,sc)][_mask,:]
                _seg_idx+=_mask.sum()
            raster_segs[:,:,0]=np.r_[0:30]
            random_segs[:,:,0]=np.r_[0:30]
            lineoff = 0.8
            linelen = 0.2
            raster_ax.eventplot(evnts,linewidths = 0.6, linelengths = linelen, 
                         lineoffsets = lineoff, color = 'black')
            # have to do the inset axes, histogram
            wf_ax.patch.set_alpha(0.02)
            raster_snips = LineCollection(raster_segs, linewidths=0.25,
                                   colors='red', 
                                   linestyle='solid')
            rand_snips = LineCollection(random_segs, linewidths=0.25,
                                   colors='black', 
                                   linestyle='solid')
            wf_ax.add_collection(rand_snips)
            wf_ax.add_collection(raster_snips)
            wf_ax.set_xlim(0,30)
            wf_ax.set_ylim(min(raster_segs[:,:,1].flatten()),max(raster_segs[:,:,1].flatten()))
            bin_width = 0.05
            hist_ax.hist(evntsArray,bins = np.r_[-lpad:0:bin_width,0:rpad:bin_width])
            hist_ax.set_xlim(-lpad,rpad)
            raster_ax.set_xlim(-lpad,rpad)
            f.text(0.1,0.95,"w:%s,sc:%s" % (wire,sc),transform = f.transFigure)
            f.set_size_inches(5,5)
            if fndec is None:
                f.savefig(os.path.join(plt_dir,"Raster_wire%s_sc%s.pdf" % (wire,sc)))
                f.savefig(os.path.join(plt_dir,"Raster_wire%s_sc%s.png" % (wire,sc)),
                          dpi = 300,transparent=True)
            else:
                f.savefig(os.path.join(plt_dir,"Raster_%s_wire%s_sc%s.pdf" % (fndec,wire,sc)))                
                f.savefig(os.path.join(plt_dir,"Raster_%s_wire%s_sc%s.png" % (fndec,wire,sc)),
                          dpi = 300,transparent=True)                
        
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
            nm_units_here = wg[wg.TDTts.between(*times)]['SC'].nunique()
            SC_cnt = 0
            for ii,(sc,g) in enumerate(wg.groupby('SC')):
                if sc==0:
                    continue
                g_mask = g.TDTts.between(*times)
                if g_mask.sum()>0:
                    axar[i].eventplot(g[g_mask]['TDTts'].values, 
                                      lineoffsets=max(pNeu[wn-1,:])+(ii*20),linelength = 20,
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