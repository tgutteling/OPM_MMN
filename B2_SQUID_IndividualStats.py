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

#Get all files 
SQUID_epochs = [d for d in listdir(op.join(group_path,'epochs')) if '.fif' in d and 'SQUID_' in d  and not 'std-' in d and not '_eve' in d]

subs=[f[:2] for f  in SQUID_epochs]
subs=np.unique(subs)
epochs=dict()
epochs={s : {'odd':[],'std':[]} for s in subs}

#Load evoked & sort
SQUID_epochs.sort()
for f in SQUID_epochs:
    sub=f[:2]
    dat=mne.read_epochs(op.join(group_path,'epochs',f))    
    picks=mne.pick_types(dat.info,meg=True,ref_meg=False)    
    if 'odd' in f:                
        epochs[sub]['odd']=dat.pick(picks=picks)
    if 'std' in f:
        epochs[sub]['std']=dat.pick(picks=picks)
    
#########
# STATS #
#########
sig=np.zeros(len(subs))
p_vals=np.ones(len(subs))
for i,s in enumerate(subs):
    cur=epochs[s]
    
    #get adjacency (neighbors)
    adjacency, ch_names = find_ch_adjacency(cur['std'].info, ch_type="mag")

    #MRT36 gets added to the adjacency matrix, but does not exist in our data
    mis_ind=np.where([1 if c=='MRT36' else 0 for c in ch_names])[0][0]

    #remove from adjacency matrix
    tmp=adjacency.toarray()
    tmp=np.delete(tmp,mis_ind,axis=1)
    tmp=np.delete(tmp,mis_ind,axis=0)
    adjacency2=scipy.sparse.csr_matrix(tmp)

    ch_names2=[c+'-2805' for c in ch_names if c!='MRT36']

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
    # (number of conditions - 1) and the denominator (number of observations
    # - number of conditions):
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
            p_all.append(p_values[c])
            
            
    if len(p_all)==0: #no sig results
        p_vals[i]=1
    else:
        p_vals[i]=np.min(p_all)
    
save_path = op.join(group_path, 'SQUID_cluster_pvals.pkl')
pickle.dump(p_vals, open(save_path, "wb"))
print('Cluster p values saved.')

