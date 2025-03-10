#Preamble
import os.path as op
from os import listdir
import sys
import mne
import numpy as np
import autoreject as ar
from itertools import groupby
import pickle
import matplotlib.pyplot as plt 
plt.ion()

#Script folder
sys.path.append('/sps/cermep/opm/NEW_MEG/mne-python/new_meg')
import mag4health_5sens as m4he

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
sub_list=[]
for d in range(len(dirs)):      
    OPM_files=[d for d in listdir(op.join(data_path, str(dirs[d]).zfill(2))) if 'OPM' in d and 'MMN' in d] 
    if len(OPM_files)>0:
        print(str(dirs[d]).zfill(2))        
        sub_list.append(str(dirs[d]).zfill(2))
    
input_given=False
while not input_given:
    inp = input('Select Subject: ') #Ask for user input    
    if inp.isdecimal() and np.any(np.isin(dirs,int(inp))):
        print('Subject '+inp+' selected')
        subject=inp.zfill(2)        
        input_given=True
    else:
        print('Incorrect input, please try again')

print('Retreiving OPM sessions..')    
OPM_files=[d for d in listdir(op.join(data_path, subject)) if 'OPM' in d and 'MMN' in d and op.isdir(op.join(data_path, subject,d))]
print('Found ' +str(len(OPM_files))+ ' OPM data directories')
OPM_files.sort()


ses_nr=[d[-1:] for d in OPM_files]
ses_nr=np.sort(ses_nr)
runs=[]                              
for i,s in enumerate(ses_nr):
    runs.append(OPM_files[i][:-1] +s)
    
#AUDIO triggers:
#    '1' - normal audio
#    '2' - oddball

if subject=='10':
    runs=[runs[1]] #ignore session 1 of sub10

all_raw=[]
for i_run in runs:
    print('Loading ' +i_run)
    file_name=op.join(data_path, subject, i_run, 'data', 'single_matrix.mat') 
    raw=m4he.read_mag4health_raw(file_name)    
    
    events, event_dict=mne.events_from_annotations(raw)      
    
    if len(events)>100: #ignore short datasets (aborted sessions)
        
        #Let's remove periods without stimulation events (startup, end, breaks)
        break_annots = mne.preprocessing.annotate_break(
        raw=raw,
        events=events,
        min_break_duration=9,  # consider segments of at least 5 s duration
        t_start_after_previous=4,  # buffer time after last event, carefull of edge effects
        t_stop_before_next=4  # stop annotation 4 s before beginning of next one
        )
        raw.set_annotations(raw.annotations + break_annots)  #Mark breaks in raw data
        
        #Mark _z direction channels as bad
        bads=mne.pick_channels_regexp(raw.info['ch_names'],'^.*z')
        raw.info['bads'].extend([raw.info['ch_names'][x] for x in bads])
        
        #Remove spikes from data
        #Reject by amplitude, use standard deviation of the raw data
        raw_tmp=raw.copy().resample(500).filter(l_freq=1, h_freq=100)    
        ref_chans=mne.pick_channels_regexp(raw.info['ch_names'],'^ZC*')
        exclude=bads+ref_chans
        goods=list(set(range(0,len(raw.ch_names)))-set(exclude))        
        dat,times=raw_tmp.copy().crop(tmin=15,tmax=raw_tmp.times[-1]-15).get_data(picks=goods,reject_by_annotation='omit',return_times=True) #raw data, just for setting threshold                        
        n_sds=5 #how many SD to set as threshold
        spike_annots, spike_bads = mne.preprocessing.annotate_amplitude(
            raw_tmp,
            peak=np.median(dat)+np.std(dat)*n_sds
            )        
        raw.set_annotations(raw.annotations + spike_annots) #add these annotations to the raw data
        
        
        #spikes get removed fine, but slower high amplitude artifacts do not get picked up
        #This is a first pass rejection to remove gross artifacts.
        raw_tmp.annotations.onset=raw_tmp.annotations.onset-raw_tmp.first_time #see note                
        raw_tmp.set_annotations(raw_tmp.annotations + spike_annots) #add these annotations to the raw data
        dat,times=raw_tmp.get_data(picks=goods,reject_by_annotation='NaN',return_times=True) #get raw data and times
        dat_crop=raw_tmp.copy().crop(tmin=15,tmax=raw_tmp.times[-1]-15).get_data(picks=goods,reject_by_annotation='omit')
        n_sds=5 #how many SD to set as threshold
        thres=np.median(dat_crop)+np.std(dat_crop)*n_sds #set the rejection threshold using SD
        bad=np.abs(dat)>thres #create bool array of threshold violations
        bad=[bad[:,i].any() for i in range(bad.shape[1])] #collapse across channels
        #Now place a 20ms sliding window in the true ranges 
        n_samp = round(raw_tmp.info['sfreq']*.01)    
        i=0
        while i<len(bad):
            if bad[i]:
                if i < n_samp:
                    bad[:i+n_samp] = np.full(len(bad[:i+n_samp]),True)
                    i+=n_samp+1                
                elif i+n_samp > len(bad):
                    bad[i-n_samp:] = np.full(len(bad[i-n_samp:]),True)                
                    break
                else:
                    bad[i-n_samp:i+n_samp] = np.full(len(bad[i-n_samp:i+n_samp]),True)                
                    i+=n_samp+1
            else:
                i+=1                        
    
        #To create annotations from this, we need to know onset times and durations        
        onset=[]
        duration=[]
        description=[]
        i = 0
        for val, g in groupby(bad):
            l=len(list(g))
            if val:       
                onset.append(times[i]) 
                if i+l>=len(times):
                    duration.append(times[-1]-times[i])
                else:
                    duration.append(times[i+l]-times[i])
                description.append('BAD_amplitude')
            i += l
        
        #Create annotations and add to data
        amplitude_annots=mne.Annotations(onset=onset, duration=duration,description=description)
        raw.annotations.onset=raw.annotations.onset-raw.first_time #see note        
        raw.set_annotations(raw.annotations + amplitude_annots)
        
        all_raw.append(raw) #add annotated raw session to the whole
    
raw=mne.concatenate_raws(all_raw, on_mismatch='warn')    

#get events from concatenated RAW
events, event_dict=mne.events_from_annotations(raw)

#Recode events such that standard sounds preceding an oddbal are coded differently.
odd_nr=event_dict['CODE2\n']
events[:,2][np.where(events[:,2]==odd_nr)[0]-1]=6
event_dict['std_preOdd'] = 6

#save event structure for trial order reconstruction
print('Saving events')
ev_file=op.join(data_path,'group','MMN',subject+'_OPM_eve.fif')
mne.write_events(ev_file, events,overwrite=True)    

#select true standard tone event
event_ids=np.unique(events[:,2])
event_count=[np.sum(events[:,2]==d) for d in event_ids]
std_id=event_ids[np.array(event_count).argsort()][-1:][0]
std_code=[s for s in event_dict.keys() if event_dict[s]==std_id]

#put in sensor locations
#load standard channel positions (previously measured)
with open(op.join(data_path,'group','MMN','ch_pos.pkl'), "rb") as f:    
    ch_pos=pickle.load(f)
    
rename=dict()
suff=['x','y','z']
for s in suff:
    rename['ZF2_' +s]='FZ2_' +s;
    rename['ZC2_' +s]='CZ2_' +s;

raw.rename_channels(rename)  
    
for ch in raw.info['chs']:
    pos=ch_pos[ch['ch_name']]
    if ch['ch_name'][-1]=='z':
        pos[0]=pos[0]+0.005
    if ch['ch_name'][-1]=='x':
        pos[0]=pos[0]-0.005
    ch['loc']=pos
       
#Subject has bad channel 
if subject=='18':
    raw.info['bads']=['FZ2_x','FZ2_y']+raw.info['bads']

raw.info['bads']=[]
picks_x=mne.pick_channels_regexp(raw.info['ch_names'],'^.*x')
picks_y=mne.pick_channels_regexp(raw.info['ch_names'],'^.*y')
picks_z=mne.pick_channels_regexp(raw.info['ch_names'],'^.*z')
ref_chans=['CZ2_x','CZ2_y','CZ2_z']
picks_x=[c for c in picks_x if raw.ch_names[c] not in ref_chans]
picks_y=[c for c in picks_y if raw.ch_names[c] not in ref_chans]
picks_z=[c for c in picks_z if raw.ch_names[c] not in ref_chans]

#filter raw data
print('filtering raw data')
raw_filt=raw.copy().notch_filter(freqs=line_freq).filter(l_freq=highpass, h_freq=lowpass, picks=picks_x+picks_y)
raw_filt=raw_filt.filter(l_freq=highpass, h_freq=10, picks=picks_z) # bandpass
raw_filt=raw_filt.filter(l_freq=highpass, h_freq=5, picks=ref_chans) # bandpass

#resample to final sampling rate
raw_filt.resample(final_samp_rate)

#Rolling correlation
from sklearn.linear_model import LinearRegression
tw=2000 #timweindow in datapoints to consider
stepsize=100 #only calculate coefficients per x steps (in datapoints)
ref_dat=raw_filt.copy().pick(picks=ref_chans).get_data()
chan_dat=raw_filt.copy().pick(picks=picks_x+picks_y+picks_z).get_data() #remove refs
datapoints=np.shape(ref_dat)[1]
coefs=np.zeros((np.shape(chan_dat)[0],4,len(range(0,datapoints,stepsize))))
tails=round(tw/2)
#loop over t (datapoints)
for c in range(np.shape(chan_dat)[0]):
    print('Processing channel ' +str(c+1)+ '/' +str(np.shape(chan_dat)[0]))
    for i,t in enumerate(range(0,datapoints,stepsize)):    
        if (t>round(tw/2)) and (t<datapoints-round(tw/2)):
            ref_tmp=(ref_dat[:,t-tails:t+tails].T)
            chan_tmp=(chan_dat[c,t-tails:t+tails])
            reg = LinearRegression().fit(ref_tmp,chan_tmp)
            coefs[c,:,i]=np.append(reg.coef_,reg.intercept_)            

chan_regr=np.zeros(np.shape(chan_dat))    
for c in range(np.shape(chan_dat)[0]):
    #upsample coefficients 
    coefs_interp=np.zeros((4,datapoints))
    for s in range(np.shape(coefs)[1]):
        coefs_interp[s,]=np.interp(np.array(range(datapoints)),np.array(range(0,datapoints,stepsize)),coefs[c,s,])    

    #apply regression
    chan_regr[c,]=(chan_dat[c,]-(np.sum((coefs_interp[0:3,]*ref_dat),axis=0)))
    
raw_regr=raw_filt.copy().pick(picks=picks_x+picks_y+picks_z)
raw_regr._data=chan_regr  

#Create datasets for ICA
raw_ICA=raw_regr.copy().resample(500)

#ICA 
ica = mne.preprocessing.ICA(method='fastica',random_state=42)
ica.fit(raw_ICA)
if not use_previous_rejection:
    ica.plot_sources(raw_ICA)

    #for comparison, use epoched data    
    events, event_dict=mne.events_from_annotations(raw_regr) #reinitialize events
    event_id={'CODE1\n' : event_dict['CODE1\n'],'CODE2\n' : event_dict['CODE2\n']} #Trigger
    tmp_epochs=mne.Epochs(raw_regr,events,event_id=event_id,tmin=tmin,tmax=tmax,preload=True)
    tmp_epochs.resample(500)
    ica.plot_sources(tmp_epochs)


#Mark  bad ICA components
ICA_reject = {
    '01' : [1,2,3],
    '02' : [1,5,8],
    '04' : [0,2,7],
    '06' : [4,5,6],
    '07' : [3,5,7],
    '08' : [1,4,6,7,9],
    '09' : [3,6,9],
    '10' : [0,3,5,9],
    '14' : [1,2,6,7],
    '15' : [3,7,8,9],
    '16' : [2,3,4,5,8],
    '17' : [1,3,7],
    '18' : [0,2,6], #poor dataquality, needs manual marking of raw + manual epoch rejection
    '19' : [1,4,8],
    '21' : [6,7,8],
    '22' : [0,6,7,9,11],
    '24' : [1,2,5,10]
    }


if use_previous_rejection:
    raw_clean=ica.apply(raw_regr,exclude=ICA_reject[subject])    
else:
    raw_clean=ica.apply(raw_regr)     

#final filter    
raw_clean=raw_clean.filter(l_freq=None, h_freq=lowpass_final)

## EPOCH
print('Creating epochs')
events, event_dict=mne.events_from_annotations(raw_clean) #reinitialize events
odd_nr=event_dict['CODE2\n']
events[:,2][np.where(events[:,2]==odd_nr)[0]-1]=6
event_dict['std_preOdd'] = 6
event_id_std={std_code[0] : event_dict[std_code[0]]} #standard tone, except the ones before oddball
event_id_stdPreOdd={'std_preOdd' : event_dict['std_preOdd']} #Standard tone before oddball
event_id_odd={'CODE2\n' : event_dict['CODE2\n']} #Oddball tone
event_id_all={std_code[0] : event_dict[std_code[0]],'std_preOdd' : event_dict['std_preOdd'],'CODE2\n' : event_dict['CODE2\n']} #all, for rejection threshold

#Cut epochs
epochs_std=mne.Epochs(raw_clean,events,event_id=event_id_std,tmin=tmin,tmax=tmax,preload=True)
epochs_stdPreOdd=mne.Epochs(raw_clean,events,event_id=event_id_stdPreOdd,tmin=tmin,tmax=tmax,preload=True)
epochs_odd=mne.Epochs(raw_clean,events,event_id=event_id_odd,tmin=tmin,tmax=tmax,preload=True)
epochs_all=mne.Epochs(raw_clean,events,event_id=event_id_all,tmin=tmin,tmax=tmax,preload=True)

#baselining
epochs_std.apply_baseline(baseline=(bl_min,bl_max))   
epochs_stdPreOdd.apply_baseline(baseline=(bl_min,bl_max))   
epochs_odd.apply_baseline(baseline=(bl_min,bl_max))   
epochs_all.apply_baseline(baseline=(bl_min,bl_max))   

#Autoreject
if use_autoreject:    
    print('Using autoreject to discard artifactual epochs')        
    rejectTHRES = ar.get_rejection_threshold(epochs_all, decim=2,random_state=42,ch_types='mag') #get AR threshold
    print('Threshold: ' +str(rejectTHRES['mag']))
    drop_std=epochs_std.copy().drop_bad(reject=rejectTHRES,verbose='WARNING') #check resulting rejection
    print('Standard tone - Portion of data kept: ' +str(len(drop_std)/np.shape(epochs_std)[0]*100)+ '%')        
    epochs_std=epochs_std.drop_bad(reject=rejectTHRES,verbose='WARNING') #check resulting reject
    
    drop_stdPreOdd=epochs_stdPreOdd.copy().drop_bad(reject=rejectTHRES,verbose='WARNING') #check resulting rejection
    print('Standard before Oddball tone - Portion of data kept: ' +str(len(drop_stdPreOdd)/np.shape(epochs_stdPreOdd)[0]*100)+ '%')        
    epochs_stdPreOdd=epochs_stdPreOdd.drop_bad(reject=rejectTHRES,verbose='WARNING') #check resulting reject
    
    drop_odd=epochs_odd.copy().drop_bad(reject=rejectTHRES,verbose='WARNING') #check resulting rejection
    print('Oddball tone - Portion of data kept: ' +str(len(drop_odd)/np.shape(epochs_odd)[0]*100)+ '%')        
    epochs_odd=epochs_odd.drop_bad(reject=rejectTHRES,verbose='WARNING') #check resulting reject

#save
print('Saving epochs..')
epochs_std.save(op.join(data_path,'group','MMN','epochs',subject+'_OPM_std-epo.fif'),overwrite=True)    
epochs_odd.save(op.join(data_path,'group','MMN','epochs',subject+'_OPM_odd-epo.fif'),overwrite=True)    
epochs_stdPreOdd.save(op.join(data_path,'group','MMN','epochs',subject+'_OPM_stdPreOdd-epo.fif'),overwrite=True)    
print('Done.')    

#Create evoked
evoked_std=epochs_std.average() 
evoked_odd=epochs_odd.average() 
evoked_stdPreOdd=epochs_stdPreOdd.average() 

#save
print('Saving evoked..')
evoked_std.save(op.join(data_path,'group','MMN','evoked',subject+'_OPM_std-ave.fif'),overwrite=True)
evoked_odd.save(op.join(data_path,'group','MMN','evoked',subject+'_OPM_odd-ave.fif'),overwrite=True)
evoked_stdPreOdd.save(op.join(data_path,'group','MMN','new_params','evoked',subject+'_OPM_stdPreOdd-ave.fif'),overwrite=True)
print('Done.')

print('Finished subject ' +subject)