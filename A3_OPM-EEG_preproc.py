# Preprocessing of EEG data recorded concurrently with OPM

#Preamble
import os.path as op
from os import listdir
import mne
import numpy as np
import autoreject as ar
import matplotlib.pyplot as plt 
plt.ion()

#Task parameters 
tmin,tmax =-0.2, 0.41 #time before and after trigger. Tones are spaced 610ms apart
bl_min,bl_max=-0.2,0 #baseline start and end (None for either first or last sample)
highpass,lowpass = 2,45 #bandpass frequencies
highpass_final,lowpass_final=2,20 #after preproc, but before epoching, apply more stingent filter
line_freq=[50,60,100,150] #notch filter frequency(/ies)
final_samp_rate=1000 #Output sampling rate
use_previous_rejection=1 #load rejection parameters from previous dataset, for consistency when reanalysing with different parameters
use_autoreject=1   

#define file name
data_path =  '/sps/cermep/opm/NEW_MEG/HV'

#Get datasets
dirs = [int(d) for d in listdir(data_path) if op.isdir(op.join(data_path, d)) and d.isnumeric()]
dirs.sort()

#Let user select subject
print('Available Subjects: ')

for d in range(len(dirs)):      
    ds_files=[d for d in listdir(op.join(data_path, str(dirs[d]).zfill(2))) if '.ds' in d and  'MEG' in d and 'MMN' in d]    
    if len(ds_files)>0:
        print(str(dirs[d]).zfill(2))        
    
input_given=False
while not input_given:
    inp = input('Select Subject: ') #Ask for user input    
    if inp.isdecimal() and np.any(np.isin(dirs,int(inp))):
        print('Subject '+inp+' selected')
        subject=inp.zfill(2)        
        input_given=True
    else:
        print('Incorrect input, please try again')

print('Retreiving sessions..') 
print('Loading data')
ds_files=[d for d in listdir(op.join(data_path, subject)) if '.ds' in d and  'MEG' in d and 'MMN' in d]
ds_files.sort()

#make sure data is in the right order
ses_nr=[int(d[-5:-3]) for d in ds_files]
ses_nr=np.sort(ses_nr)
runs=[]                              
for s in range(len(ses_nr)):
    runs.append(ds_files[0][:-5] +str(ses_nr[s]).zfill(2) +'.ds')               
    
#AUDIO triggers:
#    '1' - normal audio
#    '2' - oddball

#We have to correct the EEG channel names
rename = {     
     'EEG001-2800': 'Fp1',
     'EEG002-2800': 'Fp2',
     'EEG003-2800': 'AFz',
     'EEG004-2800': 'F1', 
     'EEG005-2800': 'F2',
     'EEG006-2800': 'FC5',
     'EEG007-2800': 'FCz',
     'EEG008-2800': 'FC6',
     'EEG009-2800': 'C1',
     'EEG010-2800': 'C2',
     'EEG011-2800': 'TP9',
     'EEG012-2800': 'TP10',
     'EEG013-2800': 'P7',
     'EEG014-2800': 'P8',
     }

st_1020=mne.channels.make_standard_montage('standard_1020')         

#load and append data
#EEG data is recorded by the SQUID-MEG system
all_raw=[]
for i_run in runs:
    print('Loading ' +i_run)
    file_name=op.join(data_path, subject, i_run) 
    
    raw = mne.io.read_raw_ctf(file_name, preload=True)    
    raw.rename_channels(rename)
    raw.set_montage(st_1020, match_alias=rename,on_missing='ignore')
    
    if len(raw.ch_names) < 100: #ignore SQUID+EEG datasets    
        
        if subject=='02':
            raw.resample(1200)
        
        #define event IDs
        events, event_dict=mne.events_from_annotations(raw)        
    
        #mark break sections
        break_annots = mne.preprocessing.annotate_break(
        raw=raw,
        events=events,
        min_break_duration=9,  # consider segments of at least 5 s duration
        t_start_after_previous=4,  # buffer time after last event, carefull of edge effects
        t_stop_before_next=4  # stop annotation 4 s before beginning of next one
        )
        raw.set_annotations(raw.annotations + break_annots)  #Mark breaks in raw data
              
        
        all_raw.append(raw) #add annotated raw session to the whole
            
raw=mne.concatenate_raws(all_raw, on_mismatch='warn')    

#get events from concatenated RAW
events, event_dict=mne.events_from_annotations(raw)
       
#Recode events such that standard sounds preceding an oddbal are coded differently.
odd_id=event_dict['2']
events[:,2][np.where(events[:,2]==odd_id)[0]-1]=6
event_dict['std_preOdd'] = 6

#save event structure for trial order reconstruction
print('Saving events')
ev_file=op.join(data_path,'group','MMN','epochs',subject+'_OPMEEG_eve.fif')
mne.write_events(ev_file, events,overwrite=True)    

#select true standard tone event
event_ids=np.unique(events[:,2])
event_count=[np.sum(events[:,2]==d) for d in event_ids]
std_id=event_ids[np.array(event_count).argsort()][-1:][0]
std_code=[s for s in event_dict.keys() if event_dict[s]==std_id]

#Split data in MEG and EEG data
raw_eeg=raw.copy().pick_types(meg=False,eeg=True,ref_meg=True)
if subject=='04':
    picks_eog=['EEG064-2800']
else:
    picks_eog=['EEG063-2800','EEG064-2800']

raw_eog=raw.copy().pick_channels(picks_eog)
picks_eeg=[c for c in raw_eeg.ch_names if c not in picks_eog]
raw_eeg=raw_eeg.pick_channels(picks_eeg) #remove eog from EEG

#filter data
raw_eog_filt=raw_eog.copy().notch_filter(freqs=line_freq).filter(l_freq=highpass, h_freq=lowpass)    
raw_eeg_filt=raw_eeg.copy().notch_filter(freqs=line_freq).filter(l_freq=highpass, h_freq=lowpass)

#Create datasets for ICA
raw_eeg_ICA=raw_eeg_filt.copy().resample(500)
    
#Manually inspect EEG for bad channels
#raw_eeg_ICA.plot()    

#MARK bad channel(s)
bad_channels = {
    '01' : [15],
    '02' : [15],
    '04' : [15,62],
    '06' : [15],
    '07' : [15],
    '08' : [2,15],
    '09' : [15],
    '10' : [15],
    '14' : [15],
    '15' : [15],
    '16' : [15,62],
    '17' : [15],
    '18' : [2,15],
    '19' : [15],
    '21' : [15],
    '22' : [15],
    '24' : [15]
    }
    
#if use_previous_rejection:
bads = [rename['EEG0' +str(d).zfill(2)+ '-2800'] for d in bad_channels[subject] if d<15]+['EEG0' +str(d).zfill(2)+ '-2800' for d in bad_channels[subject] if d>14]
    
raw_eeg_ICA.info['bads']=bads
raw_eeg_filt.info['bads']=bads            

# ICA
ica_eeg = mne.preprocessing.ICA(method='fastica',random_state=42)
ica_eeg.fit(raw_eeg_ICA)
if not use_previous_rejection:
    ica_eeg.plot_sources(raw_eeg_ICA)
    ica_eeg.plot_components()

    raw_ep=raw_eeg_filt.copy()
    event_id={'1' : event_dict['1'], '2' : event_dict['2']} #Trigger
    tmp_epochs=mne.Epochs(raw_ep,events,event_id=event_id,tmin=tmin,tmax=tmax,preload=True)
    tmp_epochs.resample(500)
    ica_eeg.plot_sources(tmp_epochs)

#BAD EEG ICA components
ICA_eeg_reject = {
    '01' : [0,1,3,5],
    '02' : [0,12],
    '04' : [0,1,7,11,13],
    '06' : [0,1,6,8,11],
    '07' : [0,1,6,7,8],
    '08' : [0,1,7,8,11],
    '09' : [0,1,2,7,11,13],
    '10' : [0,1,2,7,8,12],
    '14' : [0,1,2,3,5,7],
    '15' : [0,1,3],
    '16' : [1,2,11],
    '17' : [0,1,5,10],
    '18' : [0,1,3,4],
    '19' : [0,2,4],
    '21' : [0,1,2],
    '22' : [0,1,2,6],
    '24' : [0,1,3,4,8]
    }

if use_previous_rejection:
    raw_eeg_clean=ica_eeg.apply(raw_eeg_filt,exclude=ICA_eeg_reject[subject])    
else:
    raw_eeg_clean=ica_eeg.apply(raw_eeg_filt)   

raw_eeg_clean.add_channels([raw_eog_filt])

#Bandpass filter
print('Wide Bandpass filter')
raw_eeg_filt=raw_eeg_clean.filter(l_freq=None, h_freq=lowpass_final)

## EPOCH
#get events
print('Creating epochs')
event_id_std={std_code[0] : event_dict[std_code[0]]} #Standard tone (exl pre-odd)
event_id_stdPreOdd={'std_preOdd' : event_dict['std_preOdd']} #Standard Tone  preceding Odd
event_id_odd={'2' : event_dict['2']} #Trigger
event_id_all={std_code[0] : event_dict[std_code[0]],'std_preOdd' : event_dict['std_preOdd'],'2' : event_dict['2']} #all, for rejection threshold

#Cut epochs
eeg_epochs_std=mne.Epochs(raw_eeg_filt,events,event_id=event_id_std,tmin=tmin,tmax=tmax,preload=True)
eeg_epochs_stdPreOdd=mne.Epochs(raw_eeg_filt,events,event_id=event_id_stdPreOdd,tmin=tmin,tmax=tmax,preload=True)
eeg_epochs_odd=mne.Epochs(raw_eeg_filt,events,event_id=event_id_odd,tmin=tmin,tmax=tmax,preload=True)
eeg_epochs_all=mne.Epochs(raw_eeg_filt,events,event_id=event_id_all,tmin=tmin,tmax=tmax,preload=True)

#now resample to output sampling rate
print('Resampling.')
eeg_epochs_std.resample(final_samp_rate)
eeg_epochs_stdPreOdd.resample(final_samp_rate)
eeg_epochs_odd.resample(final_samp_rate)
eeg_epochs_all.resample(final_samp_rate)

#Correct eye movement artifacts using regression
print('Correcting for eye movements')
eeg_epochsTMP=eeg_epochs_all.copy().subtract_evoked()
if subject=='04' or subject=='16':
    picks_eog=['EEG064-2800']
else:
    picks_eog=['EEG063-2800','EEG064-2800']
eeg_epochsTMP.set_eeg_reference('average')

#fit model
eeg_model_EOG = mne.preprocessing.EOGRegression(picks='eeg',picks_artifact=picks_eog).fit(eeg_epochsTMP)

#apply model
eeg_epochs_std.set_eeg_reference('average')
eeg_epochs_stdPreOdd.set_eeg_reference('average')
eeg_epochs_odd.set_eeg_reference('average')
eeg_epochs_all.set_eeg_reference('average')
eeg_epochs_std=eeg_model_EOG.apply(eeg_epochs_std)
eeg_epochs_stdPreOdd=eeg_model_EOG.apply(eeg_epochs_stdPreOdd)
eeg_epochs_odd=eeg_model_EOG.apply(eeg_epochs_odd)
eeg_epochs_all=eeg_model_EOG.apply(eeg_epochs_all)

#drop eog
print('Removing EOG from data')
eeg_epochs_std=eeg_epochs_std.pick_channels(picks_eeg)
eeg_epochs_stdPreOdd=eeg_epochs_stdPreOdd.pick_channels(picks_eeg)
eeg_epochs_odd=eeg_epochs_odd.pick_channels(picks_eeg)
eeg_epochs_all=eeg_epochs_all.pick_channels(picks_eeg)

#baselining
print('Baselining')
eeg_epochs_std.apply_baseline(baseline=(bl_min,bl_max))   
eeg_epochs_stdPreOdd.apply_baseline(baseline=(bl_min,bl_max))   
eeg_epochs_odd.apply_baseline(baseline=(bl_min,bl_max))   
eeg_epochs_all.apply_baseline(baseline=(bl_min,bl_max))   

#Autoreject
#EEG
if use_autoreject:    
    print('EEG: Using autoreject to discard artifactual epochs')        
    rejectTHRES = ar.get_rejection_threshold(eeg_epochs_all, decim=2,random_state=42,ch_types='eeg') #get AR threshold
    print('Threshold: ' +str(rejectTHRES['eeg']))
    
    drop_std=eeg_epochs_std.copy().drop_bad(reject=rejectTHRES,verbose='WARNING') #check resulting rejection
    print('EEG Std - Portion of data kept: ' +str(len(drop_std)/np.shape(eeg_epochs_std)[0]*100)+ '%')        
    eeg_epochs_std=eeg_epochs_std.drop_bad(reject=rejectTHRES,verbose='WARNING') #check resulting rejection
    
    drop_stdPreOdd=eeg_epochs_stdPreOdd.copy().drop_bad(reject=rejectTHRES,verbose='WARNING') #check resulting rejection
    print('EEG StdPreOdd - Portion of data kept: ' +str(len(drop_stdPreOdd)/np.shape(eeg_epochs_stdPreOdd)[0]*100)+ '%')        
    eeg_epochs_stdPreOdd=eeg_epochs_stdPreOdd.drop_bad(reject=rejectTHRES,verbose='WARNING') #check resulting rejection
    
    drop_odd=eeg_epochs_odd.copy().drop_bad(reject=rejectTHRES,verbose='WARNING') #check resulting rejection
    print('EEG Odd - Portion of data kept: ' +str(len(drop_odd)/np.shape(eeg_epochs_odd)[0]*100)+ '%')        
    eeg_epochs_odd=eeg_epochs_odd.drop_bad(reject=rejectTHRES,verbose='WARNING') #check resulting rejection
        
#save
print('Saving epochs..')
eeg_epochs_std.save(op.join(data_path,'group','MMN','epochs',subject+'_OPMEEG_std-epo.fif'),overwrite=True)    
eeg_epochs_stdPreOdd.save(op.join(data_path,'group','MMN','epochs',subject+'_OPMEEG_stdPreOdd-epo.fif'),overwrite=True)    
eeg_epochs_odd.save(op.join(data_path,'group','MMN','epochs',subject+'_OPMEEG_odd-epo.fif'),overwrite=True)    
print('Done.')    

#Create evoked
eeg_evoked_std=eeg_epochs_std.average() 
eeg_evoked_stdPreOdd=eeg_epochs_stdPreOdd.average() 
eeg_evoked_odd=eeg_epochs_odd.average() 


#save
print('Saving evoked..')
eeg_evoked_std.save(op.join(data_path,'group','MMN','evoked',subject+'_OPMEEG_std-ave.fif'),overwrite=True)
eeg_evoked_stdPreOdd.save(op.join(data_path,'group','MMN','evoked',subject+'_OPMEEG_stdPreOdd-ave.fif'),overwrite=True)
eeg_evoked_odd.save(op.join(data_path,'group','MMN','evoked',subject+'_OPMEEG_odd-ave.fif'),overwrite=True)
print('Done.')

print('Finished subject ' +subject)