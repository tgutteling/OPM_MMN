# Here we collect all preprocessed data from the MMN experiment and create group results

#Preamble
import os.path as op
import sys
import mne
import numpy as np
from os import listdir

sys.path.append('/sps/cermep/opm/NEW_MEG/mne-python/new_meg')
group_path = '/sps/cermep/opm/NEW_MEG/HV/group/MMN/'

import matplotlib.pyplot as plt 
plt.ion()

#Get all files 
SQUID_evoked = [d for d in listdir(op.join(group_path,'evoked')) if '.fif' in d and 'SQUID_' in d]
OPMEEG_evoked = [d for d in listdir(op.join(group_path,'evoked')) if '.fif' in d and 'OPMEEG' in d]
SQUIDEEG_evoked = [d for d in listdir(op.join(group_path,'evoked')) if '.fif' in d and 'SQUIDEEG' in d]
OPM_evoked = [d for d in listdir(op.join(group_path,'evoked')) if '.fif' in d and 'OPM_' in d]

subs=[f[:2] for f  in SQUID_evoked]
subs=np.unique(subs)
evoked=dict()
evoked={s : {'SQUID' : {'odd':[],'std':[]}, 'OPMEEG' : {'odd':[],'std':[]}, 'SQUIDEEG' : {'odd':[],'std':[]}, 'OPM' : {'odd':[],'std':[]}} for s in subs}


#Load evoked & sort
SQUID_evoked.sort()
for f in SQUID_evoked:
    sub=f[:2]
    if not 'std-' in f:                
        dat=mne.read_evokeds(op.join(group_path,'evoked',f),condition=0)    
    if 'odd' in f:                
        evoked[sub]['SQUID']['odd']=dat
    if 'stdPreOdd' in f:        
        evoked[sub]['SQUID']['std']=dat
    
OPMEEG_evoked.sort()
for f in OPMEEG_evoked:
    sub=f[:2]
    if not 'std-' in f:                
        dat=mne.read_evokeds(op.join(group_path,'evoked',f),condition=0)    
    if 'odd' in f:                
        evoked[sub]['OPMEEG']['odd']=dat
    if 'stdPreOdd' in f:        
        evoked[sub]['OPMEEG']['std']=dat
        
SQUIDEEG_evoked.sort()
for f in SQUIDEEG_evoked:
    sub=f[:2]
    if not 'std-' in f:                
        dat=mne.read_evokeds(op.join(group_path,'evoked',f),condition=0)    
    if 'odd' in f:                
        evoked[sub]['SQUIDEEG']['odd']=dat
    if 'stdPreOdd' in f:        
        evoked[sub]['SQUIDEEG']['std']=dat        
        
OPM_evoked.sort()
for f in OPM_evoked:
    if not 'std-' in f:                
        sub=f[:2]
        dat=mne.read_evokeds(op.join(group_path,'evoked',f),condition=0)    
        if 'odd' in f:                
            evoked[sub]['OPM']['odd']=dat
        if 'stdPreOdd' in f:        
            evoked[sub]['OPM']['std']=dat


#bad channel 'interpolation' for subject 18 (12th sub), channels FZ2_x/y
#we'll replace it with the average of all other participants.
std_FZ2_x_av=[]
std_FZ2_y_av=[]
odd_FZ2_x_av=[]
odd_FZ2_y_av=[]
for s in evoked.keys():
    if s != '18':
        std_FZ2_x_av.append(evoked[s]['OPM']['std'].data[mne.pick_channels_regexp(evoked[s]['OPM']['std'].info['ch_names'],'FZ2_x'),])
        std_FZ2_y_av.append(evoked[s]['OPM']['std'].data[mne.pick_channels_regexp(evoked[s]['OPM']['std'].info['ch_names'],'FZ2_y'),])
        odd_FZ2_x_av.append(evoked[s]['OPM']['odd'].data[mne.pick_channels_regexp(evoked[s]['OPM']['odd'].info['ch_names'],'FZ2_x'),])
        odd_FZ2_y_av.append(evoked[s]['OPM']['odd'].data[mne.pick_channels_regexp(evoked[s]['OPM']['odd'].info['ch_names'],'FZ2_y'),])

evoked['18']['OPM']['std']._data[mne.pick_channels_regexp(evoked[s]['OPM']['std'].info['ch_names'],'FZ2_x'),]=np.mean(std_FZ2_x_av,axis=0)
evoked['18']['OPM']['std']._data[mne.pick_channels_regexp(evoked[s]['OPM']['std'].info['ch_names'],'FZ2_y'),]=np.mean(std_FZ2_y_av,axis=0)
evoked['18']['OPM']['odd']._data[mne.pick_channels_regexp(evoked[s]['OPM']['odd'].info['ch_names'],'FZ2_x'),]=np.mean(odd_FZ2_x_av,axis=0)
evoked['18']['OPM']['odd']._data[mne.pick_channels_regexp(evoked[s]['OPM']['odd'].info['ch_names'],'FZ2_y'),]=np.mean(odd_FZ2_y_av,axis=0)

ev_std_SQUID=[]
ev_odd_SQUID=[]
ev_std_OPMEEG=[]
ev_odd_OPMEEG=[]
ev_std_SQUIDEEG=[]
ev_odd_SQUIDEEG=[]
ev_std_OPM=[]
ev_odd_OPM=[]
for s in evoked.keys():
    ev_std_SQUID.append(evoked[s]['SQUID']['std'])
    ev_odd_SQUID.append(evoked[s]['SQUID']['odd'])
    rm_eeg=mne.pick_channels_regexp(evoked[s]['OPMEEG']['std'].ch_names,'EEG')
    evoked[s]['OPMEEG']['std'].drop_channels([evoked[s]['OPMEEG']['std'].ch_names[r] for r in rm_eeg])
    evoked[s]['OPMEEG']['odd'].drop_channels([evoked[s]['OPMEEG']['odd'].ch_names[r] for r in rm_eeg])
    ev_std_OPMEEG.append(evoked[s]['OPMEEG']['std'])    
    ev_odd_OPMEEG.append(evoked[s]['OPMEEG']['odd'])
    rm_eeg=mne.pick_channels_regexp(evoked[s]['SQUIDEEG']['std'].ch_names,'EEG')
    evoked[s]['SQUIDEEG']['std'].drop_channels([evoked[s]['SQUIDEEG']['std'].ch_names[r] for r in rm_eeg])
    evoked[s]['SQUIDEEG']['odd'].drop_channels([evoked[s]['SQUIDEEG']['odd'].ch_names[r] for r in rm_eeg])
    ev_std_SQUIDEEG.append(evoked[s]['SQUIDEEG']['std'])
    ev_odd_SQUIDEEG.append(evoked[s]['SQUIDEEG']['odd'])
    ev_std_OPM.append(evoked[s]['OPM']['std'])
    ev_odd_OPM.append(evoked[s]['OPM']['odd'])

ga_STD_SQUID=mne.grand_average(ev_std_SQUID)
ga_ODD_SQUID=mne.grand_average(ev_odd_SQUID)
ga_STD_OPMEEG=mne.grand_average(ev_std_OPMEEG)
ga_ODD_OPMEEG=mne.grand_average(ev_odd_OPMEEG)
ga_STD_SQUIDEEG=mne.grand_average(ev_std_SQUIDEEG)
ga_ODD_SQUIDEEG=mne.grand_average(ev_odd_SQUIDEEG)
ga_STD_OPM=mne.grand_average(ev_std_OPM,interpolate_bads=False,drop_bads=False)
ga_ODD_OPM=mne.grand_average(ev_odd_OPM,interpolate_bads=False,drop_bads=False)
ga_STD_OPM.info['bads']=[]
ga_ODD_OPM.info['bads']=[]

#Save
ga_STD_SQUID.save(op.join(group_path,'averages','ga_STD_SQUID-ave.fif'),overwrite=True)
ga_ODD_SQUID.save(op.join(group_path,'averages','ga_ODD_SQUID-ave.fif'),overwrite=True)
ga_STD_OPMEEG.save(op.join(group_path,'averages','ga_STD_OPMEEG-ave.fif'),overwrite=True)
ga_ODD_OPMEEG.save(op.join(group_path,'averages','ga_ODD_OPMEEG-ave.fif'),overwrite=True)
ga_STD_SQUIDEEG.save(op.join(group_path,'averages','ga_STD_SQUIDEEG-ave.fif'),overwrite=True)
ga_ODD_SQUIDEEG.save(op.join(group_path,'averages','ga_ODD_SQUIDEEG-ave.fif'),overwrite=True)
ga_STD_OPM.save(op.join(group_path,'averages','ga_STD_OPM-ave.fif'),overwrite=True)
ga_ODD_OPM.save(op.join(group_path,'averages','ga_ODD_OPM-ave.fif'),overwrite=True)
 