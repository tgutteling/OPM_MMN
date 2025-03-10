# Figure 3 - Direct time course comparison between OPM, SQUID & EEG

#Preamble
import os.path as op
import mne
import numpy as np
from copy import deepcopy
import seaborn as sns

group_path = '/sps/cermep/opm/NEW_MEG/HV/group/MMN/'

import matplotlib.pyplot as plt 
plt.ion()

#load grand average data
av_dir='/sps/cermep/opm/NEW_MEG/HV/group/MMN/averages'
ga_STD_SQUID=mne.read_evokeds(op.join(av_dir,'ga_STD_SQUID-ave.fif'),condition=0)    
ga_ODD_SQUID=mne.read_evokeds(op.join(av_dir,'ga_ODD_SQUID-ave.fif'),condition=0)    
ga_STD_OPMEEG=mne.read_evokeds(op.join(av_dir,'ga_STD_OPMEEG-ave.fif'),condition=0)    
ga_ODD_OPMEEG=mne.read_evokeds(op.join(av_dir,'ga_ODD_OPMEEG-ave.fif'),condition=0)    
ga_STD_SQUIDEEG=mne.read_evokeds(op.join(av_dir,'ga_STD_SQUIDEEG-ave.fif'),condition=0)    
ga_ODD_SQUIDEEG=mne.read_evokeds(op.join(av_dir,'ga_ODD_SQUIDEEG-ave.fif'),condition=0)    
ga_STD_OPM=mne.read_evokeds(op.join(av_dir,'ga_STD_OPM-ave.fif'),condition=0)    
ga_ODD_OPM=mne.read_evokeds(op.join(av_dir,'ga_ODD_OPM-ave.fif'),condition=0)    
    
#Let's find channels closest to the OPM sensor locations
picks_y=mne.pick_channels_regexp(ga_STD_OPM.info['ch_names'],'^[FL].*y')
chan_match_SQUID=dict()
for s in picks_y:    
    opm_loc=ga_STD_OPM.info['chs'][s]['loc'][0:3]-[0, 0, .04] #slight correction for OPM location being a bit high
    dist_SQ=[]
    ind_SQ=[]
    for m in range(len(ga_STD_SQUID.info['chs'])):
        if ga_STD_SQUID.info['chs'][m]['kind']==1: #standard channel
            tmp=opm_loc-ga_STD_SQUID.info['chs'][m]['loc'][0:3]
            dist_SQ.append(np.sqrt(np.dot(tmp.T, tmp)))
            ind_SQ.append(m)
    match_ind=ind_SQ[np.where(dist_SQ==np.min(dist_SQ))[0][0]]
    match_name=ga_STD_SQUID.ch_names[match_ind]
    chan_match_SQUID[ga_STD_OPM.info['chs'][s]['ch_name']]=match_name
    
#chan_match_SQUID - {'FZ2_y': 'MRF41-2805',  'LC11_y': 'MLC51-2805', 'LT15_y': 'MLF46-2805', 'LT34_y': 'MLC25-2805'}

#same for EEG..
chan_match_eeg=dict()
for s in picks_y:    
    opm_loc=ga_STD_OPM.info['chs'][s]['loc'][0:3]-[0, 0, .04]
    dist_EG=[]
    ind_EG=[]
    for m in range(len(ga_STD_OPMEEG.info['chs'])):
        if ga_STD_OPMEEG.info['chs'][m]['kind']==2: #standard channel
            tmp=opm_loc-ga_STD_OPMEEG.info['chs'][m]['loc'][0:3]
            dist_EG.append(np.sqrt(np.dot(tmp.T, tmp)))
            ind_EG.append(m)
    match_ind=ind_EG[np.where(dist_EG==np.min(dist_EG))[0][0]]
    match_name=ga_STD_OPMEEG.ch_names[match_ind]
    chan_match_eeg[ga_STD_OPM.info['chs'][s]['ch_name']]=match_name

#chan_match_eeg {'FZ2_y': 'AFz', 'LC11_y': 'C1', 'LT15_y': 'TP9', 'LT34_y': 'P7'}

########
# PLOT #
########

fig,ax=plt.subplots(4,1)    
fig.set_figheight(12)
fig.set_figwidth(8)
fig.subplots_adjust(hspace=0.3)
MMN_SQUID=deepcopy(ga_STD_SQUID)
MMN_SQUID._data=ga_ODD_SQUID.data-ga_STD_SQUID.data
MMN_SQUID.apply_baseline(baseline=(-0.2, 0))
MMN_OPMEEG=deepcopy(ga_STD_OPMEEG)
MMN_OPMEEG._data=ga_ODD_OPMEEG.data-ga_STD_OPMEEG.data
MMN_OPMEEG.apply_baseline(baseline=(-0.2, 0))
MMN_SQUIDEEG=deepcopy(ga_STD_SQUIDEEG)
MMN_SQUIDEEG._data=ga_ODD_SQUIDEEG.data-ga_STD_SQUIDEEG.data
MMN_SQUIDEEG.apply_baseline(baseline=(-0.2, 0))
MMN_OPM=deepcopy(ga_STD_OPM)
MMN_OPM._data=ga_ODD_OPM.data-ga_STD_OPM.data
MMN_OPM.apply_baseline(baseline=(-0.2, 0))
chans_plot=['FZ2_y','LT34_y','LT15_y','LC11_y']
for i,s in enumerate(chans_plot):        

    tmp_dat=MMN_SQUID.copy().pick(picks=chan_match_SQUID[s]).data    
    bl_sd=np.std(tmp_dat[:,MMN_SQUID.times<0])    
    tmp_dat=tmp_dat/bl_sd
    ax[i].plot(MMN_SQUID.times,tmp_dat.T)
    
    tmp_dat=MMN_OPM.copy().pick(picks=s).data
    bl_sd=np.std(tmp_dat[:,MMN_OPM.times<0])    
    tmp_dat=tmp_dat/bl_sd
    if s=='LT34_y':
        ax[i].plot(MMN_OPM.times,tmp_dat.T)
    else:
        ax[i].plot(MMN_OPM.times,-tmp_dat.T)
    
    tmp_dat=MMN_OPM.copy().pick(picks=s[:-1]+'x').data
    bl_sd=np.std(tmp_dat[:,MMN_OPM.times<0])    
    tmp_dat=tmp_dat/bl_sd
    if s=='LC11_y':
        ax[i].plot(MMN_OPM.times,tmp_dat.T)
    else:
        ax[i].plot(MMN_OPM.times,-tmp_dat.T)        
        
    tmp_dat1=MMN_SQUIDEEG.copy().pick(picks=chan_match_eeg[s]).data
    bl_sd=np.std(tmp_dat1[:,MMN_SQUIDEEG.times<0])    
    tmp_dat1=tmp_dat1/bl_sd
    tmp_dat2=MMN_OPMEEG.copy().pick(picks=chan_match_eeg[s]).data
    bl_sd=np.std(tmp_dat2[:,MMN_OPMEEG.times<0])
    tmp_dat2=tmp_dat2/bl_sd
    tmp_dat=np.average(np.vstack((tmp_dat1,tmp_dat2)),axis=0)
    ax[i].plot(MMN_SQUIDEEG.times,tmp_dat.T)
           
    ax[i].set_ylim([-20, 20])
    ax[i].set_title(s[:-2])     
    ax[i].set_ylabel('Normalized MMN')

ax[3].set_xlabel('Time (s)')    

labels=['SQUID','OPM$_{radial}$','OPM$_{tangential}$','EEG']
ax[0].legend(labels,loc='upper left')

for n in range(len(ax)):
    ax[n].set_xlim([-0.1,0.4])
    ax[n].spines.bottom.set_bounds(-.1,.4)
    ax[n].spines.left.set_bounds(-20,20)            
    ax[n].vlines(0,-20,20,color='k',linestyle='--')    
    ax[n].hlines(0,-0.1,0.4,color='k')
    ax[n].spines.right.set_visible(False)
    ax[n].spines.top.set_visible(False)
        
    
fig.savefig(op.join(group_path,'Grand_average_MMN_SensorCompare_perLoc_PolarityEqual_avEEG_blSD.pdf'))    
fig.savefig(op.join(group_path,'Grand_average_MMN_SensorCompare_perLoc_PolarityEqual_avEEG_blSD.png'))    

#Timecourse comparison
chan='LT34'
OPMY_av=MMN_OPM.copy().pick(picks=[chan+ '_y']).get_data()
OPMX_av=MMN_OPM.copy().pick(picks=[chan+ '_x']).get_data()
SQUID_av=MMN_SQUID.copy().pick(picks=chan_match_SQUID[chan+ '_y']).get_data()
OPMEEG_av=MMN_OPMEEG.copy().pick(picks=chan_match_eeg[chan+ '_y']).get_data()
SQUIDEEG_av=MMN_SQUIDEEG.copy().pick(picks=chan_match_eeg[chan+ '_y']).get_data()

chan_data=np.vstack((SQUID_av,OPMY_av,OPMX_av,SQUIDEEG_av,OPMEEG_av))
corr=np.corrcoef(chan_data)

labels=['SQUID-MEG','OPM-MEG$_{radial}$','OPM-MEG$_{tangential}$','EEG$_{SQUID}$','EEG$_{OPM}$']

fig, ax = plt.subplots(1, 1, figsize=(5,2))            
sns.heatmap(np.abs(corr), annot=corr,fmt=".3f",xticklabels=labels,yticklabels=labels,cmap='Greys',vmin=0,vmax=1,cbar=True,ax=ax)#cmap='rocket',
ax.xaxis.set_tick_params(rotation=0)
ax.set_title(chan)
    
fig.savefig(op.join(group_path,'Timecourse_Correlation_cbar_' +chan+ '.pdf'))    
fig.savefig(op.join(group_path,'Timecourse_Correlation_cbar_' +chan+ '.png'))    

#For repsorting: peak extraction for lateral sensors (LT34 equivalent)
SQUID_st=ga_STD_SQUID.copy().pick(picks=chan_match_SQUID['LT34_y'])
SQUID_odd=ga_ODD_SQUID.copy().pick(picks=chan_match_SQUID['LT34_y'])
SQUID_st.apply_baseline(baseline=(-0.2, 0))
SQUID_odd.apply_baseline(baseline=(-0.2, 0))
m1_st=SQUID_st.get_peak(tmin=0,tmax=.12)[1]
m1_odd=SQUID_odd.get_peak(tmin=0,tmax=.12)[1]
m2_st=SQUID_st.get_peak(tmin=0.1,tmax=.22)[1]
m2_odd=SQUID_odd.get_peak(tmin=0.1,tmax=.22)[1]
SQUID_peaks=[m1_st,m1_odd,m2_st,m2_odd]
SQUID_avPeaks=[np.average([m1_st,m1_odd]),np.average([m2_st,m2_odd])]

#OPMY
OPMY_st=ga_STD_OPM.copy().pick(picks='LT34_y')
OPMY_odd=ga_ODD_OPM.copy().pick(picks='LT34_y')
OPMY_st.apply_baseline(baseline=(-0.2, 0))
OPMY_odd.apply_baseline(baseline=(-0.2, 0))
m1_st=OPMY_st.get_peak(tmin=0,tmax=.12)[1]
m1_odd=OPMY_odd.get_peak(tmin=0,tmax=.12)[1]
m2_st=OPMY_st.get_peak(tmin=0.1,tmax=.22)[1]
m2_odd=OPMY_odd.get_peak(tmin=0.1,tmax=.22)[1]
OPMY_peaks=[m1_st,m1_odd,m2_st,m2_odd]
OPMY_avPeaks=[np.average([m1_st,m1_odd]),np.average([m2_st,m2_odd])]

#OPMX
OPMX_st=ga_STD_OPM.copy().pick(picks='LT34_x')
OPMX_odd=ga_ODD_OPM.copy().pick(picks='LT34_x')
OPMX_st.apply_baseline(baseline=(-0.2, 0))
OPMX_odd.apply_baseline(baseline=(-0.2, 0))
m1_st=OPMX_st.get_peak(tmin=0,tmax=.09)[1]
m1_odd=OPMX_odd.get_peak(tmin=0,tmax=.09)[1]
m2_st=OPMX_st.get_peak(tmin=0.1,tmax=.15)[1]
m2_odd=OPMX_odd.get_peak(tmin=0.1,tmax=.15)[1]
OPMX_peaks=[m1_st,m1_odd,m2_st,m2_odd]
OPMX_avPeaks=[np.average([m1_st,m1_odd]),np.average([m2_st,m2_odd])]

#EEG
SQUIDEEG_st=ga_STD_SQUIDEEG.copy().pick(picks=chan_match_eeg['LT34_y'])
SQUIDEEG_odd=ga_ODD_SQUIDEEG.copy().pick(picks=chan_match_eeg['LT34_y'])
OPMEEG_st=ga_STD_OPMEEG.copy().pick(picks=chan_match_eeg['LT34_y'])
OPMEEG_odd=ga_ODD_OPMEEG.copy().pick(picks=chan_match_eeg['LT34_y'])
EEG_st=SQUIDEEG_st.copy()
EEG_odd=SQUIDEEG_odd.copy()
EEG_st._data=(SQUIDEEG_st.data+OPMEEG_st.data)/2
EEG_odd._data=(SQUIDEEG_odd.data+OPMEEG_odd.data)/2
EEG_st.apply_baseline(baseline=(-0.2, 0))
EEG_odd.apply_baseline(baseline=(-0.2, 0))
m1_st=EEG_st.get_peak(tmin=0,tmax=.09)[1]
m1_odd=EEG_odd.get_peak(tmin=0,tmax=.09)[1]
m2_st=EEG_st.get_peak(tmin=0.1,tmax=.15)[1]
m2_odd=EEG_odd.get_peak(tmin=0.1,tmax=.15)[1]
EEG_peaks=[m1_st,m1_odd,m2_st,m2_odd]
EEG_avPeaks=[np.average([m1_st,m1_odd]),np.average([m2_st,m2_odd])]

print('SQUID Peaks: ' +str(np.round((SQUID_avPeaks[0]*1000),0))+ 'ms / ' +str(np.round((SQUID_avPeaks[1]*1000),0))+ 'ms')
print('OPM radial Peaks: ' +str(np.round((OPMY_avPeaks[0]*1000),0))+ 'ms / ' +str(np.round((OPMY_avPeaks[1]*1000),0))+ 'ms')
print('OPM tang1 Peaks: ' +str(np.round((OPMX_avPeaks[0]*1000),0))+ 'ms / ' +str(np.round((OPMX_avPeaks[1]*1000),0))+ 'ms')
print('EEG Peaks: ' +str(np.round((EEG_avPeaks[0]*1000),0))+ 'ms / ' +str(np.round((EEG_avPeaks[1]*1000),0))+ 'ms')




