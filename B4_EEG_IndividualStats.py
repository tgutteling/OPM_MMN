# Here we use all individual data to check for a significant within-subject MMN (data used in figure 4)

#Preamble
import os.path as op
import mne
import numpy as np
from os import listdir
from mne.channels import find_ch_adjacency
from mne.stats import spatio_temporal_cluster_test
import scipy
import pickle

group_path = '/sps/cermep/opm/NEW_MEG/HV/group/MMN/'

import matplotlib.pyplot as plt 
plt.ion()

EEG_ses={s:[] for s in ['SQUIDEEG','OPMEEG']}
for i,EEG_type in enumerate([1,2]):

    #Get all files 
    if EEG_type==1:
        EEG_epochs = [d for d in listdir(op.join(group_path,'epochs')) if '.fif' in d and 'SQUIDEEG_' in d  and not 'std-' in d]
        s_type='SQUIDEEG'
    
    if EEG_type==2:
        EEG_epochs = [d for d in listdir(op.join(group_path,'epochs')) if '.fif' in d and 'OPMEEG_' in d  and not 'std-' in d and not '_eve' in d]
        s_type='OPMEEG'        
    
    subs=[f[:2] for f  in EEG_epochs]
    subs=np.unique(subs)
    epochs=dict()
    epochs={s : {'odd':[],'std':[]} for s in subs}
    
    #Load evoked & sort
    EEG_epochs.sort()
    for f in EEG_epochs:
        if EEG_type<3:    
                sub=f[:2]
                dat=mne.read_epochs(op.join(group_path,'epochs',f))    
                picks=mne.pick_types(dat.info,eeg=True,meg=False,ref_meg=False)
                dat.pick(picks=picks)
                if 'odd' in f:                
                    epochs[sub]['odd']=dat
                if 'std' in f:        
                    epochs[sub]['std']=dat
        else:
            sub=f[:2]
            dat=mne.read_epochs(op.join(group_path,'epochs',f))    
            picks=mne.pick_types(dat.info,eeg=True,meg=False,ref_meg=False)
            dat.pick(picks=picks)
            if 'odd' in f:                
                epochs[sub]['odd'].append(dat)
            if 'std' in f:        
                epochs[sub]['std'].append(dat)
    
    EEG_ses[s_type]=epochs


#########
# STATS #
#########
ses_sig={ses:[] for ses in EEG_ses.keys()}
ses_p_vals={ses:[] for ses in EEG_ses.keys()}
ses_sig_inds={ses:[] for ses in EEG_ses.keys()}

for ses in EEG_ses.keys():
    sig=np.zeros(len(subs))
    pval=np.zeros(len(subs))
    sig_inds={i :[] for i in subs}
    for i,s in enumerate(subs):
                
        cur=EEG_ses[ses][s]
        
        #get adjacency (neighbors)
        adjacency, ch_names = find_ch_adjacency(cur['std'].info, ch_type="eeg")        
        
        #P8-C1 connection is wrong, remove
        #also FP1-TP9
        tmp=adjacency.toarray()
        P8_ind=[d for d in range(len(ch_names)) if ch_names[d]=='P8']
        C1_ind=[d for d in range(len(ch_names)) if ch_names[d]=='C1']
        if (len(P8_ind)>0) & (len(C1_ind)>0):
            tmp[P8_ind[0],C1_ind[0]]=0
            tmp[C1_ind[0],P8_ind[0]]=0
        
        #also Fp1-TP9
        FP1_ind=[d for d in range(len(ch_names)) if ch_names[d]=='Fp1']
        TP9_ind=[d for d in range(len(ch_names)) if ch_names[d]=='TP9']
        if (len(FP1_ind)>0) & (len(TP9_ind)>0):
            tmp[FP1_ind[0],TP9_ind[0]]=0
            tmp[TP9_ind[0],FP1_ind[0]]=0
            
        #remove P7-P8 connection (Across hemisheres)        
        P8_ind=[d for d in range(len(ch_names)) if ch_names[d]=='P8']
        P7_ind=[d for d in range(len(ch_names)) if ch_names[d]=='P7']
        if (len(P8_ind)>0) & (len(P7_ind)>0):
            tmp[P8_ind[0],P7_ind[0]]=0
            tmp[P7_ind[0],P8_ind[0]]=0
        
        #FC5-Fp1
        FC5_ind=[d for d in range(len(ch_names)) if ch_names[d]=='FC5']
        FC6_ind=[d for d in range(len(ch_names)) if ch_names[d]=='FC6']
        FP2_ind=[d for d in range(len(ch_names)) if ch_names[d]=='Fp2']
        if (len(FC5_ind)>0) & (len(FP1_ind)>0):
            tmp[FC5_ind[0],FP1_ind[0]]=0
            tmp[FP1_ind[0],FC5_ind[0]]=0
        if (len(FC6_ind)>0) & (len(FP2_ind)>0):
            tmp[FC6_ind[0],FP2_ind[0]]=0
            tmp[FP2_ind[0],FC6_ind[0]]=0
        
        #FC5-TP9 / FC6-TP10
        TP10_ind=[d for d in range(len(ch_names)) if ch_names[d]=='TP10']
        if (len(FC5_ind)>0) & (len(TP9_ind)>0):
            tmp[FC5_ind[0],TP9_ind[0]]=0
            tmp[TP9_ind[0],FC5_ind[0]]=0
        if (len(FC6_ind)>0) & (len(TP10_ind)>0):
            tmp[FC6_ind[0],TP10_ind[0]]=0
            tmp[TP10_ind[0],FC6_ind[0]]=0
        
        #FC5-P7 / FC6-P8
        if (len(FC5_ind)>0) & (len(P7_ind)>0):
            tmp[FC5_ind[0],P7_ind[0]]=0
            tmp[P7_ind[0],FC5_ind[0]]=0
        if (len(FC6_ind)>0) & (len(P8_ind)>0):
            tmp[FC6_ind[0],P8_ind[0]]=0
            tmp[P8_ind[0],FC6_ind[0]]=0
        
        #C1-P7 / C2-P8
        C2_ind=[d for d in range(len(ch_names)) if ch_names[d]=='C2']
        if (len(C2_ind)>0) & (len(P7_ind)>0):
            tmp[C1_ind[0],P7_ind[0]]=0
            tmp[P7_ind[0],C1_ind[0]]=0
        if (len(C2_ind)>0) & (len(P8_ind)>0):        
            tmp[C2_ind[0],P8_ind[0]]=0
            tmp[P8_ind[0],C2_ind[0]]=0
        
        
        adjacency2=scipy.sparse.csr_matrix(tmp)        
        
        #data
        x1 = cur['std'].get_data() 
        x2 = cur['odd'].get_data() 
        X=[x1,x2]
        # the dimensions are as expected for the cluster permutation test:
        # n_epochs × n_times × n_channels
        X = [np.transpose(x, (0, 2, 1)) for x in X]
        
        # We are running an F test, so we look at the upper tail
        # see also: https://stats.stackexchange.com/a/73993
        tail = 1
        
        # We want to set a critical test statistic (here: F), to determine when
        # clusters are being formed. Using Scipy's percent point function of the F
        # distribution, we can conveniently select a threshold that corresponds to
        # some alpha level that we arbitrarily pick.        
        alpha_cluster_forming = 0.01
        
        # For an F test we need the degrees of freedom for the numerator    
        n_conditions = 2
        n_observations = len(X[0])
        dfn = n_conditions - 1
        dfd = n_observations - n_conditions
        
        # Note: we calculate 1 - alpha_cluster_forming to get the critical value
        # on the right tail
        f_thresh = scipy.stats.f.ppf(1 - alpha_cluster_forming, dfn=dfn, dfd=dfd)
        
        # run the cluster based permutation analysis
        cluster_stats = spatio_temporal_cluster_test(
            X,
            n_permutations=1000,
            threshold=f_thresh,
            tail=tail,
            n_jobs=None,
            buffer_size=None,
            adjacency=adjacency2,
        )
        F_obs, clusters, p_values, _ = cluster_stats
        
        p_accept = 0.05
        good_cluster_inds = np.where(p_values < p_accept)[0]
        
        p_all=[]
        for c in range(len(p_values)):
            time_inds, space_inds = np.squeeze(clusters[c])
            time_inds=np.unique(time_inds)
            if (cur['std'].times[time_inds[0]]>0) & (np.shape((np.where((cur['std'].times[time_inds]>.1) & (cur['std'].times[time_inds]<.2))))[1]>1):
                if p_values[c]<p_accept:
                    sig[i]=1
                    sig_inds[s].append(np.unique(space_inds))
                p_all.append(p_values[c])
                
                
        if len(p_all)==0: #no sig results            
            pval[i]=1
        else:
            pval[i]=np.min(p_all)
                
    print('Individual results: ' +str(int(np.sum(sig)))+ '/' +str(len(sig))+ ' significant')
    ses_sig[ses]=sig
    ses_p_vals[ses]=pval
    ses_sig_inds[ses]=sig_inds

save_path = op.join(group_path, 'EEG_cluster_pvals.pkl')
pickle.dump(ses_p_vals, open(save_path, "wb"))
print('Cluster p values saved.')
