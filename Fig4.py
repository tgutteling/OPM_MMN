# Here we collect all preprocessed epoch data from the MMN experiment and look at SNR
# As there are systemic differences in the amount of trials rejected for MEG and OPM, which affects SNR, we will compensate for this

#Preamble
import os.path as op
import mne
import numpy as np
from os import listdir
import pickle
import seaborn as sns
import matplotlib.pyplot as plt 
plt.ion()

#Set up data for import
group_path = '/sps/cermep/opm/NEW_MEG/HV/group/MMN/'

recompute=0

if recompute:
    iter=1000 #number of iterations (bootstrapping)
    #Get subject files and sort by type
    fifs_epochs = [d for d in listdir(op.join(group_path,'epochs')) if '.fif' in d]
    OPMstd_files, OPModd_files, MEGstd_files, MEGodd_files,OPMEEGstd_files, OPMEEGodd_files, SQUIDEEGstd_files, SQUIDEEGodd_files=[],[],[],[],[],[],[],[]
    OPM_subs, MEG_subs,OPMEEG_subs, SQUIDEEG_subs=np.array([]),np.array([]),np.array([]),np.array([])
    for d in range(len(fifs_epochs)):      
        if (fifs_epochs[d][0:2]).isdecimal() and 'SQUID_stdPreOdd-epo' in fifs_epochs[d]: #MEG epochs object
            MEGstd_files=MEGstd_files+[fifs_epochs[d]]
            MEG_subs=np.append(MEG_subs,int(fifs_epochs[d][:2]))
        elif (fifs_epochs[d][0:2]).isdecimal() and 'OPM_stdPreOdd-epo' in fifs_epochs[d]: #OPM epochs object
            OPMstd_files=OPMstd_files+[fifs_epochs[d]]
            OPM_subs=np.append(OPM_subs,int(fifs_epochs[d][:2]))
        elif (fifs_epochs[d][0:2]).isdecimal() and 'OPMEEG_stdPreOdd-epo' in fifs_epochs[d]: #OPMEEG epochs object
            OPMEEGstd_files=OPMEEGstd_files+[fifs_epochs[d]]
            OPMEEG_subs=np.append(OPMEEG_subs,int(fifs_epochs[d][:2]))        
        elif (fifs_epochs[d][0:2]).isdecimal() and 'SQUIDEEG_stdPreOdd-epo' in fifs_epochs[d]: #SQUIDEEG epochs object
            SQUIDEEGstd_files=SQUIDEEGstd_files+[fifs_epochs[d]]
            SQUIDEEG_subs=np.append(SQUIDEEG_subs,int(fifs_epochs[d][:2]))                
        elif (fifs_epochs[d][0:2]).isdecimal() and 'SQUID_odd' in fifs_epochs[d]: #OPM epochs object
            MEGodd_files=MEGodd_files+[fifs_epochs[d]]
            MEG_subs=np.append(MEG_subs,int(fifs_epochs[d][:2]))
        elif (fifs_epochs[d][0:2]).isdecimal() and 'OPM_odd' in fifs_epochs[d]: #OPM epochs object
            OPModd_files=OPModd_files+[fifs_epochs[d]]
            OPM_subs=np.append(OPM_subs,int(fifs_epochs[d][:2]))        
        elif (fifs_epochs[d][0:2]).isdecimal() and 'OPMEEG_odd' in fifs_epochs[d]: #OPMEEG epochs object
            OPMEEGodd_files=OPMEEGodd_files+[fifs_epochs[d]]
            OPMEEG_subs=np.append(OPMEEG_subs,int(fifs_epochs[d][:2]))                
        elif (fifs_epochs[d][0:2]).isdecimal() and 'SQUIDEEG_odd' in fifs_epochs[d]: #OPMEEG epochs object
            SQUIDEEGodd_files=SQUIDEEGodd_files+[fifs_epochs[d]]
            SQUIDEEG_subs=np.append(SQUIDEEG_subs,int(fifs_epochs[d][:2]))                
            
    MEG_subs=np.unique(MEG_subs)
    OPM_subs=np.unique(OPM_subs)
    OPMEEG_subs=np.unique(OPMEEG_subs)
    SQUIDEEG_subs=np.unique(SQUIDEEG_subs)
        
    #Load data and calculate SNR
    com_subs1=np.intersect1d(MEG_subs,OPM_subs) #Do this only for common subjects
    com_subs2=np.intersect1d(OPMEEG_subs,SQUIDEEG_subs) #Do this only for common subjects
    com_subs=np.intersect1d(com_subs1,com_subs2)
    
    #we need to know the number of trials per subject/modality
    cnt=0
    MEG_nr_trials=np.zeros([len(com_subs),2])
    OPMEEG_nr_trials=np.zeros([len(com_subs),2])
    SQUIDEEG_nr_trials=np.zeros([len(com_subs),2])
    OPM_nr_trials=np.zeros([len(com_subs),2])
    for d in com_subs:       
        dat_MEGstd=mne.read_epochs(op.join(group_path,'epochs',str(int(d)).zfill(2)+'_SQUID_stdPreOdd-epo.fif'))
        dat_MEGodd=mne.read_epochs(op.join(group_path,'epochs',str(int(d)).zfill(2)+'_SQUID_odd-epo.fif'))
        dat_OPMEEGstd=mne.read_epochs(op.join(group_path,'epochs',str(int(d)).zfill(2)+'_OPMEEG_stdPreOdd-epo.fif'))
        dat_OPMEEGodd=mne.read_epochs(op.join(group_path,'epochs',str(int(d)).zfill(2)+'_OPMEEG_odd-epo.fif'))
        dat_SQUIDEEGstd=mne.read_epochs(op.join(group_path,'epochs',str(int(d)).zfill(2)+'_SQUIDEEG_stdPreOdd-epo.fif'))
        dat_SQUIDEEGodd=mne.read_epochs(op.join(group_path,'epochs',str(int(d)).zfill(2)+'_SQUIDEEG_odd-epo.fif'))
        dat_OPMstd=mne.read_epochs(op.join(group_path,'epochs',str(int(d)).zfill(2)+'_OPM_stdPreOdd-epo.fif'))
        dat_OPModd=mne.read_epochs(op.join(group_path,'epochs',str(int(d)).zfill(2)+'_OPM_odd-epo.fif'))
        MEG_nr_trials[cnt,0]=np.shape(dat_MEGstd)[0]
        MEG_nr_trials[cnt,1]=np.shape(dat_MEGodd)[0]
        OPMEEG_nr_trials[cnt,0]=np.shape(dat_OPMEEGstd)[0]
        OPMEEG_nr_trials[cnt,1]=np.shape(dat_OPMEEGodd)[0]
        SQUIDEEG_nr_trials[cnt,0]=np.shape(dat_SQUIDEEGstd)[0]
        SQUIDEEG_nr_trials[cnt,1]=np.shape(dat_SQUIDEEGodd)[0]
        OPM_nr_trials[cnt,0]=np.shape(dat_OPMstd)[0]
        OPM_nr_trials[cnt,1]=np.shape(dat_OPModd)[0]
        cnt+=1
    
    
    #save values
    nTrials = {"SQUID": MEG_nr_trials,
               "SQUIDEEG": SQUIDEEG_nr_trials,
               "OPMEEG": OPMEEG_nr_trials,
              "OPM": OPM_nr_trials
              }
    
    save_path = op.join(group_path, "nTrials.pkl")
    pickle.dump(nTrials, open(save_path, "wb"))
    print('Rejection rates saved.')  
    
    trials_min=np.min(np.hstack((MEG_nr_trials,OPM_nr_trials,OPMEEG_nr_trials,SQUIDEEG_nr_trials)),axis=1)
    
    #MEG    
    #SQUID SNR order:  ['MLC25-2805', 'MLC51-2805', 'MLF46-2805', 'MRF41-2805']
    chan_match_SQUID = {'FZ2_y': 'MRF41-2805',  'LC11_y': 'MLC51-2805', 'LT15_y': 'MLF46-2805', 'LT34_y': 'MLC25-2805'}
    SNR_MEG=np.zeros((len(com_subs),4))
    dat_cnt=0
    for d in com_subs: 
        print('Loading MEG data from subject ' +str(int(d)))      
        dat_std=mne.read_epochs(op.join(group_path,'epochs',str(int(d)).zfill(2)+'_SQUID_stdPreOdd-epo.fif'))
        dat_odd=mne.read_epochs(op.join(group_path,'epochs',str(int(d)).zfill(2)+'_SQUID_odd-epo.fif'))
        
        #reduce to relevant sensors
        picks_SQUID=['MRF41-2805', 'MLC51-2805', 'MLF46-2805', 'MLC25-2805']    
        dat_std.pick_channels(picks_SQUID)
        dat_odd.pick_channels(picks_SQUID)
        
        #cut baseline and stimulation bits
        std_stim=dat_std.copy().crop(tmin=0.1,tmax=0.25)
        odd_stim=dat_odd.copy().crop(tmin=0.1,tmax=0.25)
        std_bl=dat_std.copy().crop(tmin=-0.1,tmax=0)
        odd_bl=dat_odd.copy().crop(tmin=-0.1,tmax=0)
        
        #take minimum length
        SQUID_tr=np.min(MEG_nr_trials,axis=1)        
    
        #calculate MMN and difference baseline
        mmn=std_stim.get_data()[:int(SQUID_tr[dat_cnt]),:,:]-odd_stim.get_data()[:int(SQUID_tr[dat_cnt]),:,:]        
        bl_diff=std_bl.get_data()[:int(SQUID_tr[dat_cnt]),:,:]-odd_bl.get_data()[:int(SQUID_tr[dat_cnt]),:,:]
        
        #estimate SNR
        #peak value / std of baseline            

        if SQUID_tr[dat_cnt]<trials_min[dat_cnt]: #More OPM trials, use all MEG trials
            print('Calculating MEG SNR..')            
            signal=np.max(np.abs(np.average(mmn,axis=0)),axis=1)            
            noise=np.average((np.std(bl_diff,axis=0)/np.sqrt(np.shape(bl_diff)[0])),axis=1)            
            SNR=signal/noise
            print('Done.')
            
        else: #more MEG trials, we need to subselect and calculate SNR, repeat to average out selection bias
            print('MEG has more trials, using bootstrapping to compensate..')
            tr=int(trials_min[dat_cnt])
            
            SNR_est=np.zeros([iter,4]) 
            MEG_tr=np.arange(len(mmn))
            for i in range(0,iter):                 
                np.random.shuffle(MEG_tr) #shuffle trial nrs
                tmp_mmn=mmn[MEG_tr[0:tr],:,:]
                tmp_bl=bl_diff[MEG_tr[0:tr],:,:]            
                
                #calculate SNR            
                signal=np.max(np.abs(np.average(tmp_mmn,axis=0)),axis=1)            
                noise=np.average((np.std(tmp_bl,axis=0)/np.sqrt(np.shape(tmp_bl)[0])),axis=1)
                SNR_est[i,]=signal/noise
                
                if np.mod(i,25)==0:
                    print('Iterating ' +str(i)+ '/' +str(iter))
            
            SNR=np.average(SNR_est,axis=0)
            print('Done.')
                
        SNR_MEG[dat_cnt,:]=SNR.transpose()
        dat_cnt=dat_cnt+1
    
    #EEG    
    #chan_match_eeg {'FZ2_y': 'AFz', 'LC11_y': 'C1', 'LT15_y': 'TP9', 'LT34_y': 'P7'}
    #EEG SNR order:  ['AFz', 'C1', 'TP9', 'P7']
    SNR_OPMEEG=np.zeros((len(com_subs),4))
    dat_cnt=0
    for d in com_subs: 
        print('Loading EEG data from subject ' +str(int(d)))      
        dat_std=mne.read_epochs(op.join(group_path,'epochs',str(int(d)).zfill(2)+'_OPMEEG_stdPreOdd-epo.fif'))
        dat_odd=mne.read_epochs(op.join(group_path,'epochs',str(int(d)).zfill(2)+'_OPMEEG_odd-epo.fif'))
        
        #reduce to relevant sensors
        picks_EEG=['AFz', 'C1', 'TP9', 'P7']        
        dat_std.pick_channels(picks_EEG)
        dat_odd.pick_channels(picks_EEG)
        
        #cut baseline and stimulation bits
        std_stim=dat_std.copy().crop(tmin=0.1,tmax=0.25)
        odd_stim=dat_odd.copy().crop(tmin=0.1,tmax=0.25)
        std_bl=dat_std.copy().crop(tmin=-0.1,tmax=0)
        odd_bl=dat_odd.copy().crop(tmin=-0.1,tmax=0)
        
        #take minimum length
        EEG_tr=np.min(OPMEEG_nr_trials,axis=1)
        
    
        #calculate MMN and difference baseline
        mmn=std_stim.get_data()[:int(EEG_tr[dat_cnt]),:,:]-odd_stim.get_data()[:int(EEG_tr[dat_cnt]),:,:]        
        bl_diff=std_bl.get_data()[:int(EEG_tr[dat_cnt]),:,:]-odd_bl.get_data()[:int(EEG_tr[dat_cnt]),:,:]
        
        
        #estimate SNR
        #peak value / std of baseline                            
        if EEG_tr[dat_cnt]<trials_min[dat_cnt]: #More OPM trials, use all EEG trials
            print('Calculating OPM-EEG SNR..')            
            signal=np.max(np.abs(np.average(mmn,axis=0)),axis=1)            
            noise=np.average((np.std(bl_diff,axis=0)/np.sqrt(np.shape(bl_diff)[0])),axis=1)            
            SNR=signal/noise
            print('Done.')
        else: #more EEG trials, we need to subselect and calculate SNR, repeat to average out selection bias
            print('OPM-EEG has more trials, using bootstrapping to compensate..')
            tr=int(trials_min[dat_cnt])            
            SNR_est=np.zeros([iter,4]) 
            MEG_tr=np.arange(len(mmn))
            for i in range(0,iter):                 
                np.random.shuffle(MEG_tr) #shuffle trial nrs
                tmp_mmn=mmn[MEG_tr[0:tr],:,:]
                tmp_bl=bl_diff[MEG_tr[0:tr],:,:]            
                
                #calculate SNR            
                signal=np.max(np.abs(np.average(tmp_mmn,axis=0)),axis=1)            
                noise=np.average((np.std(tmp_bl,axis=0)/np.sqrt(np.shape(tmp_bl)[0])),axis=1)            
                SNR_est[i,]=signal/noise
                
                if np.mod(i,25)==0:
                    print('Iterating ' +str(i)+ '/' +str(iter))
            
            SNR=np.average(SNR_est,axis=0)
            print('Done.')
                
        SNR_OPMEEG[dat_cnt,:]=SNR.transpose()
        dat_cnt=dat_cnt+1
        
    SNR_SQUIDEEG=np.zeros((len(com_subs),4))
    dat_cnt=0
    for d in com_subs: 
        print('Loading SQUID-EEG data from subject ' +str(int(d)))      
        dat_std=mne.read_epochs(op.join(group_path,'epochs',str(int(d)).zfill(2)+'_SQUIDEEG_stdPreOdd-epo.fif'))
        dat_odd=mne.read_epochs(op.join(group_path,'epochs',str(int(d)).zfill(2)+'_SQUIDEEG_odd-epo.fif'))
        
        #reduce to relevant sensors
        picks_EEG=['AFz', 'C1', 'TP9', 'P7']
        dat_std.pick_channels(picks_EEG)
        dat_odd.pick_channels(picks_EEG)
        
        #cut baseline and stimulation bits
        std_stim=dat_std.copy().crop(tmin=0.1,tmax=0.25)
        odd_stim=dat_odd.copy().crop(tmin=0.1,tmax=0.25)
        std_bl=dat_std.copy().crop(tmin=-0.1,tmax=0)
        odd_bl=dat_odd.copy().crop(tmin=-0.1,tmax=0)
        
        #take minimum length
        EEG_tr=np.min(SQUIDEEG_nr_trials,axis=1)
            
        #calculate MMN and difference baseline
        mmn=std_stim.get_data()[:int(EEG_tr[dat_cnt]),:,:]-odd_stim.get_data()[:int(EEG_tr[dat_cnt]),:,:]        
        bl_diff=std_bl.get_data()[:int(EEG_tr[dat_cnt]),:,:]-odd_bl.get_data()[:int(EEG_tr[dat_cnt]),:,:]
        
        #estimate SNR
        #peak value / std of baseline                    
        if EEG_tr[dat_cnt]<trials_min[dat_cnt]: #More OPM trials, use all EEG trials
            print('Calculating SQUID-EEG SNR..')            
            signal=np.max(np.abs(np.average(mmn,axis=0)),axis=1)            
            noise=np.average((np.std(bl_diff,axis=0)/np.sqrt(np.shape(bl_diff)[0])),axis=1)            
            SNR=signal/noise
            print('Done.')
        else: #more EEG trials, we need to subselect and calculate SNR, repeat to average out selection bias
            print('SQUID-EEG has more trials, using bootstrapping to compensate..')
            tr=int(trials_min[dat_cnt])        
            SNR_est=np.zeros([iter,4]) 
            MEG_tr=np.arange(len(mmn))
            for i in range(0,iter): 
                np.random.shuffle(MEG_tr) #shuffle trial nrs
                tmp_mmn=mmn[MEG_tr[0:tr],:,:]
                tmp_bl=bl_diff[MEG_tr[0:tr],:,:]            
                
                #calculate SNR            
                signal=np.max(np.abs(np.average(tmp_mmn,axis=0)),axis=1)            
                noise=np.average((np.std(tmp_bl,axis=0)/np.sqrt(np.shape(tmp_bl)[0])),axis=1)                
                SNR_est[i,]=signal/noise
                
                if np.mod(i,25)==0:
                    print('Iterating ' +str(i)+ '/' +str(iter))
            
            SNR=np.average(SNR_est,axis=0)
            print('Done.')
                
        SNR_SQUIDEEG[dat_cnt,:]=SNR.transpose()
        dat_cnt=dat_cnt+1
        

    #OPM
    SNR_OPM=np.zeros((len(com_subs),12))
    dat_cnt=0
    templ=mne.read_epochs(op.join(group_path,'epochs',str(int(com_subs[0])).zfill(2)+'_OPM_stdPreOdd-epo.fif')) #load 'template' file for others to match
    OPM_chans=templ.ch_names
    for d in com_subs: 
        print('Loading OPM data from subject ' +str(int(d)))            
        dat_std=mne.read_epochs(op.join(group_path,'epochs',str(int(d)).zfill(2)+'_OPM_stdPreOdd-epo.fif'))
        dat_odd=mne.read_epochs(op.join(group_path,'epochs',str(int(d)).zfill(2)+'_OPM_odd-epo.fif'))
        
        #equalize channels
        [templ,dat_std,dat_odd]=mne.equalize_channels([templ,dat_std,dat_odd])        
        
        
        #cut baseline and stimulation bits
        std_stim=dat_std.copy().crop(tmin=0.1,tmax=0.25)
        odd_stim=dat_odd.copy().crop(tmin=0.1,tmax=0.25)
        std_bl=dat_std.copy().crop(tmin=-0.2,tmax=0)
        odd_bl=dat_odd.copy().crop(tmin=-0.2,tmax=0)
        
        OPM_tr=np.min(OPM_nr_trials,axis=1)
        
        #calculate MMN and difference baseline
        mmn=std_stim.get_data()[:int(OPM_tr[dat_cnt]),:,:]-odd_stim.get_data()[:int(OPM_tr[dat_cnt]),:,:]        
        bl_diff=std_bl.get_data()[:int(OPM_tr[dat_cnt]),:,:]-odd_bl.get_data()[:int(OPM_tr[dat_cnt]),:,:]    
        
        if OPM_tr[dat_cnt]<=trials_min[dat_cnt]: #More MEG trials, use all OPM trials
            print('Calculating OPM SNR..')
            signal=np.max(np.abs(np.average(mmn,axis=0)),axis=1)
            noise=np.average((np.std(bl_diff,axis=0)/np.sqrt(np.shape(bl_diff)[0])),axis=1)            
            SNR=signal/noise
            print('Done.')
        else:
            print('OPM has more trials, using bootstrapping to compensate..')
            tr=int(trials_min[dat_cnt])
            SNR_est=np.zeros([iter,12]) 
            trials=np.arange(len(mmn))        
            for i in range(0,iter):
                np.random.shuffle(trials) #shuffle trial nrs
                tmp_mmn=mmn[trials[0:tr],:,:]
                tmp_bl=bl_diff[trials[0:tr],:,:]            
                
                #calculate SNR            
                signal=np.max(np.abs(np.average(tmp_mmn,axis=0)),axis=1)            
                noise=np.average((np.std(tmp_bl,axis=0)/np.sqrt(np.shape(tmp_bl)[0])),axis=1)                
                SNR_est[i,]=signal/noise        
                
                if np.mod(i,25)==0:
                    print('Iterating ' +str(i)+ '/' +str(iter))
             
            SNR=np.average(SNR_est,axis=0)
            print('Done.')
    
        SNR_OPM[dat_cnt,:]=SNR.transpose()
        dat_cnt=dat_cnt+1     
    
    #save values
    SNR_MMN = {"SQUID": SNR_MEG,
               "SQUIDEEG": SNR_SQUIDEEG,
               "OPMEEG": SNR_OPMEEG,
              "OPM": SNR_OPM       
              }
    
    save_path = op.join(group_path, "MMN_SNR.pkl")
    pickle.dump(SNR_MMN, open(save_path, "wb"))
    print('SNR saved.')            
else:
    #load SNR values
    print('Loading SNR values')
    SNR_file = open(op.join(group_path, "MMN_SNR.pkl"),'rb')
    SNR_MMN=pickle.load(SNR_file)
    
    SNR_MEG=SNR_MMN['SQUID']
    SNR_SQUIDEEG=SNR_MMN['SQUIDEEG']
    SNR_OPMEEG=SNR_MMN['OPMEEG']
    SNR_OPM=SNR_MMN['OPM']
    

#Reorder SNRs to have the same order thoughout
#LT34 - LT15 - LC11 - FZ2
SNR_OPMEEG=SNR_OPMEEG[:,[3,2,1,0]]
SNR_SQUIDEEG=SNR_SQUIDEEG[:,[3,2,1,0]]
SNR_MEG=SNR_MEG[:,[0,2,1,3]]
SNR_OPM=SNR_OPM[:,[3,7,11,2,6,10,1,5,9,0,4,8]]
        
#Take the maximum SNR over all sensors
SNR_MEG_mx=np.max(SNR_MEG,axis=1)
SNR_OPMEEG_mx=np.max(SNR_OPMEEG,axis=1)
SNR_SQUIDEEG_mx=np.max(SNR_SQUIDEEG,axis=1)
SNR_OPM_mx_x=np.max(SNR_OPM[:,0:12:3],axis=1)
SNR_OPM_mx_y=np.max(SNR_OPM[:,1:12:3],axis=1)

comparison_labels= [
    'SQUID-MEG',
    'OPM$_{radial}$',
    'OPM$_{tangential}$',
    'EEG$_{SQUID}$',
    'EEG$_{OPM}$',    
    ]

data=np.vstack((SNR_MEG_mx,SNR_OPM_mx_y,SNR_OPM_mx_x,SNR_SQUIDEEG_mx,SNR_OPMEEG_mx)).transpose()
fig,ax=plt.subplots(1,1)
fig.set_figheight(5)
fig.set_figwidth(6)
vlp=ax.violinplot(data,showmedians=True)
ax.set_title('Signal-to-noise ratio')
ax.set_ylabel('SNR (peak signal / std baseline)')
ax.set_xlabel('Sensors')
ax.set_xticks(1+np.arange(len(comparison_labels)),labels=comparison_labels)
vlp['bodies'][0].set_facecolor('k')
vlp['bodies'][1].set_facecolor('k')
vlp['bodies'][2].set_facecolor('k')
vlp['bodies'][3].set_facecolor('k')
vlp['bodies'][4].set_facecolor('k')
vlp['cmins'].set_color('k') 
vlp['cmaxes'].set_color('k') 
vlp['cbars'].set_color('k') 
vlp['cmedians'].set_color('k') 
ax.set_ylim([0,20])
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
fig.savefig(op.join(group_path,'MMN_SNR_mx.pdf')) 
fig.savefig(op.join(group_path,'MMN_SNR_mx.png')) 

#save data as csv for analysis in JASP
data=np.vstack((SNR_MEG_mx,SNR_OPM_mx_y,SNR_OPM_mx_x,SNR_SQUIDEEG_mx,SNR_OPMEEG_mx)).T
np.savetxt(op.join(group_path,'SNR_data.csv'), data, delimiter=',')


######################
# Individual results #
######################

with open(op.join(group_path,'OPM_cluster_pvals.pkl'), "rb") as f:
    OPM_pvals = pickle.load(f)
with open(op.join(group_path,'SQUID_cluster_pvals.pkl'), "rb") as f:
    SQUID_pvals = pickle.load(f)
with open(op.join(group_path,'EEG_cluster_pvals.pkl'), "rb") as f:
    EEG_pvals = pickle.load(f)   

SQUID_sig=(SQUID_pvals<.05)*1
OPMX_sig=(OPM_pvals['x']<.05)*1
OPMY_sig=(OPM_pvals['y']<.05)*1
SQUIDEEG_sig=(EEG_pvals['SQUIDEEG']<.05)*1
OPMEEG_sig=(EEG_pvals['OPMEEG']<.05)*1

all_sig=np.vstack([SQUID_sig,OPMY_sig,OPMX_sig,SQUIDEEG_sig,OPMEEG_sig])
all_pvals=np.vstack([SQUID_pvals,OPM_pvals['y'],OPM_pvals['x'],EEG_pvals['SQUIDEEG'],EEG_pvals['OPMEEG']])

ylabels=['SQUID','OPM$_{radial}$','OPM$_{tangential}$','EEG$_{SQUID}$','EEG$_{OPM}$']
subs=['01', '02', '04', '06', '07', '08', '09', '10', '14', '15', '16', '17', '18', '19', '21', '22', '24']

cm2=[(.7,.1,0),(.8,.8,.8)]
fig, ax = plt.subplots(1, 1, figsize=(12, 4))            
sns.heatmap(all_sig, annot=all_pvals,fmt=".3f",xticklabels=subs,yticklabels=ylabels,vmin=0,vmax=.05,cmap=cm2,cbar=False,ax=ax)
ax.set_xlabel('Subject')

fig.savefig(op.join(group_path,'Individual_Results_sig.pdf'))
fig.savefig(op.join(group_path,'Individual_Results_sig.png'))    

#rejection rates
with open(op.join(group_path,'nTrials.pkl'), "rb") as f:
    nTrials = pickle.load(f)
    
#convert to data retained
pc_SQUID=np.mean((nTrials['SQUID']/224)*100,axis=1)    
pc_SQUIDEEG=np.mean((nTrials['SQUIDEEG']/224)*100,axis=1)    
pc_OPMEEG=np.mean((nTrials['OPMEEG']/224)*100,axis=1)    
pc_OPM=np.mean((nTrials['OPM']/224)*100,axis=1)    

tr_data=np.vstack((pc_SQUID,pc_OPM,pc_SQUIDEEG,pc_OPMEEG))

labels=['SQUID-MEG','OPM-MEG','EEG$_{SQUID}$','EEG$_{OPM}$']

fig, ax = plt.subplots(1, 1, figsize=(5,3))            
sns.heatmap(tr_data, annot=tr_data,fmt=".1f",xticklabels=subs,yticklabels=labels,cmap='rocket',vmin=0,vmax=100,cbar=True,ax=ax)#cmap='rocket',
    
fig.savefig(op.join(group_path,'Trials_retained.pdf'))    
fig.savefig(op.join(group_path,'Trials_retained.png'))    