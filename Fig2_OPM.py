# Figure 2 - OPM results, time window stats

#Preamble
import os.path as op
import sys
import mne
import numpy as np
from os import listdir
import scipy

sys.path.append('/sps/cermep/opm/NEW_MEG/mne-python/new_meg')
group_path = '/sps/cermep/opm/NEW_MEG/HV/group/MMN/'

import matplotlib.pyplot as plt 
plt.ion()

load_precomputed=1

#Get all files 
OPM_evoked = [d for d in listdir(op.join(group_path,'evoked')) if '.fif' in d and '_OPM_' in d  and not 'std-' in d]

#get subjects
subs=[f[:2] for f  in OPM_evoked]
subs=np.unique(subs)

#Load evoked & sort
evoked=dict()
evoked={s : {'odd':[],'std':[]} for s in subs}
OPM_evoked.sort()
for f in OPM_evoked:    
        sub=f[:2]
        dat=mne.read_evokeds(op.join(group_path,'evoked',f),condition=0)            
        if 'odd' in f:                
            evoked[sub]['odd']=dat
        if 'std' in f:        
            evoked[sub]['std']=dat
       

#create grand average
ev_std_OPM=[]
ev_odd_OPM=[]
for s in evoked.keys():
    ev_std_OPM.append(evoked[s]['std'])
    ev_odd_OPM.append(evoked[s]['odd'])

#grand average
ga_STD_OPM=mne.grand_average(ev_std_OPM,interpolate_bads=False)
ga_ODD_OPM=mne.grand_average(ev_odd_OPM,interpolate_bads=False)

#add together
evokeds=[]
evokeds.append(ga_STD_OPM)
evokeds[0].comment='std'
evokeds.append(ga_ODD_OPM)
evokeds[1].comment='odd'


mmn=evokeds[0].copy()
mmn.data=evokeds[0].data-evokeds[1].data

#########
# STATS #
#########

chans=[d for d in ev_std_OPM[0].ch_names if '_z' not in d]

tw=[.118,.196]  #left cluster SQUID  
res={c:[] for c in chans}    
times = ev_std_OPM[0].times    
for c in chans:
    ch_std=[]
    ch_odd=[]
    for s in range(len(ev_std_OPM)):
        if c not in ev_std_OPM[s].info['bads']:
            ch_std.append(ev_std_OPM[s].copy().pick(picks=c).data)
            ch_odd.append(ev_odd_OPM[s].copy().pick(picks=c).data)    
                
    toi_inds=np.where((times >= tw[0]) & (times <= tw[1]))
    tmp_std=np.squeeze(np.asarray(ch_std)[:,:,toi_inds])
    tmp_odd=np.squeeze(np.asarray(ch_odd)[:,:,toi_inds])
    res[c]=scipy.stats.ttest_rel(np.mean(tmp_std,axis=1),np.mean(tmp_odd,axis=1))
                
colors = {"odd": "crimson", "std": "steelblue"}        

for c in chans:
    sigs = ''
    sigs_b = ''
    stat, p = res[c]    
    if p < 1:
        ch_std = []
        ch_odd = []
        for s in range(len(ev_std_OPM)):
             ch_std.append(ev_std_OPM[s].copy().pick(picks=c).data)
             ch_odd.append(ev_odd_OPM[s].copy().pick(picks=c).data)

        ch_std = np.mean(np.squeeze(ch_std), axis=0)*1e15
        ch_odd = np.mean(np.squeeze(ch_odd), axis=0)*1e15
   
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        ax.plot(times, ch_std, label="standard", color=colors['std'])
        ax.plot(times, ch_odd, label="odd", color=colors['odd'])
        ax.plot(times, ch_odd - ch_std, label="MMN", color='k')
        lims = ax.get_xlim()
        ax.hlines(0, lims[0], lims[1], 'k')
        ax.set_xlim(-.1, .4)
        ax.spines.bottom.set_bounds(-.1, .4)
        lims = ax.get_ylim()
        ax.vlines(0, lims[0], lims[1], 'k', linestyle='--')
        ax.set_ylim(lims)
        low_bound = ax.get_yticks()[np.where(
            ax.get_yticks() > ax.get_ylim()[0])[0][0]]
        up_bound = ax.get_yticks()[np.where(
            ax.get_yticks() < ax.get_ylim()[1])[0][-1]]
        ax.spines.left.set_bounds(low_bound, up_bound)
        ax.set_ylabel("MEG (fT)")
        ax.set_xlabel("Time (s)")
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)
        
        if len(chans)*p < 0.05:
            print('Significant result: ' + c + ' p=' + str((len(chans)*p)))
            ax.axvspan(tw[0], tw[1], color="orange", alpha=0.3)
        else:            
            ax.axvspan(tw[0], tw[1], color='k', alpha=0.3)                 
       
        if p<.001:
            sigs='p<.001'
        else:
            sigs= 'p=' +str(round(p, 3))
            
        if (len(chans)*p)<.001:
            sigs_b='p<.001'
            ax.set_title('Channel: ' +c+ ' ' +sigs_b)
        else:
            if len(chans)*p>=1:
                sigs_b = 'p=1.0'
            else:                    
                sigs_b = 'p=' +str(round((len(chans)*p), 3))
            ax.set_title('Channel : ' + c +' ' +sigs_b+ ' (' +sigs+ ' uncorrected)')            
        
                                      
   
        ax.legend()
        
        #save   
        fig.savefig(op.join(group_path, 'Fig2_OPM_' + c + '.pdf'))
        fig.savefig(op.join(group_path, 'Fig2_OPM_' + c + '.png'))
