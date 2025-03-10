# Here we use all individual data to check for a significant within-subject MMN (data used in figure 4)

#Preamble
import os.path as op
import mne
import numpy as np
from os import listdir
import scipy
import pickle

group_path = '/sps/cermep/opm/NEW_MEG/HV/group/MMN/'

import matplotlib.pyplot as plt 
plt.ion()

#Get all files 
OPM_epochs = [d for d in listdir(op.join(group_path,'epochs')) if '.fif' in d and '_OPM_' in d  and not 'std-' in d and not '_eve' in d]

subs=[f[:2] for f  in OPM_epochs]
subs=np.unique(subs)
epochs=dict()
epochs={s : {'odd':[],'std':[]} for s in subs}

#Load evoked & sort
OPM_epochs.sort()
for f in OPM_epochs:    
        sub=f[:2]
        dat=mne.read_epochs(op.join(group_path,'epochs',f))            
        if 'odd' in f:                
            epochs[sub]['odd']=dat
        if 'std' in f:        
            epochs[sub]['std']=dat
           

#########
# STATS #
#########

tw={'early':[.118,.196]}  #left cluster SQUID  
times = epochs['01']['std'].times
OPMX_sig_tw=np.zeros(len(subs))
OPMY_sig_tw=np.zeros(len(subs))
OPMX_pvals=np.ones(len(subs))
OPMX_pvals_corr=np.ones(len(subs))
OPMY_pvals=np.ones(len(subs))
OPMY_pvals_corr=np.ones(len(subs))

for sens in ['x','y']:
    sig_tw=np.zeros(len(subs))
    for i,s in enumerate(subs):
        
        chans=[d for d in epochs[s]['std'].ch_names if '_'+sens in d]        
        res={c: {t : [] for t in tw.keys()} for c in chans}    
        
        for c in chans:
            ch_std=[]
            ch_odd=[]        
            if c not in epochs[s]['std'].info['bads']:                
                ch_std=np.squeeze(epochs[s]['std'].copy().pick(picks=c).get_data())
                ch_odd=np.squeeze(epochs[s]['odd'].copy().pick(picks=c).get_data())
                   
                min_length=np.min([np.shape(ch_std)[0],np.shape(ch_odd)[0]])
                ch_std=ch_std[:min_length,]
                ch_odd=ch_odd[:min_length,]
            
                for t in tw.keys():
                    toi_inds=np.where((times >= tw[t][0]) & (times <= tw[t][1]))
                    tmp_std=np.squeeze(np.asarray(ch_std)[:,toi_inds])
                    tmp_odd=np.squeeze(np.asarray(ch_odd)[:,toi_inds])
                    res[c][t]=scipy.stats.ttest_rel(np.mean(tmp_std,axis=1),np.mean(tmp_odd,axis=1))
                    if (res[c][t].pvalue*len(chans))<.05:
                        sig_tw[i]=1                        
                        
        sub_p=[]
        for c in res.keys():
            for t in res[c].keys():
                if len(res[c][t])>0:
                    sub_p.append(res[c][t].pvalue)
        
        if sens=='x':
            OPMX_pvals[i]=np.min(sub_p)
            OPMX_pvals_corr[i]=np.min([1,np.min(sub_p)*len(chans)])
            OPMX_sig_tw=sig_tw
        
        if sens=='y':
            OPMY_pvals[i]=np.min(sub_p)
            OPMY_pvals_corr[i]=np.min([1,np.min(sub_p)*len(chans)])
            OPMY_sig_tw=sig_tw
                
            
OPM_pvals=dict()            
OPM_pvals['x']=OPMX_pvals_corr
OPM_pvals['y']=OPMY_pvals_corr
save_path = op.join(group_path, 'OPM_cluster_pvals.pkl')
pickle.dump(OPM_pvals, open(save_path, "wb"))
print('Cluster p values saved.')                
    
         