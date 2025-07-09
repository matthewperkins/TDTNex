name = "TDTNex"
from TDTNex.TDTNexReader import TDTNex
from TDTNex.TDTNexReader import get_avg_fps_float, get_avg_fps_frac, get_r_fps, get_movie_dur, count_frames
from TDTNex.TDTNexReader import rle, trig_signal_avgsem, descritized_spike_train, BalloonProgram
from TDTNex.TDTNexReader import get_lsr_stim_blocks, tdt_ts_to_mov_ts, DeMultiPlex, sec_to_time_stamp, time_stamp_to_sec
from TDTNex.TDTNexReader import make_bursts, addSB, events_to_bursts
from TDTNex.TDTNexReader import MakeOEDataClip, MakeHLDataClip, VidDataStream, LaserDataClip, drop_bad_first_FrmN
form TDTNex.QRS_tools import rolling_RR_guess, fill_missing_QRS, make_QRS_artifact_rolling, drop_doublets

