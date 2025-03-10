# Figure 2 - EEG results, permutation cluster stats

#Preamble
import os.path as op
import sys
import mne
import numpy as np
from os import listdir
from copy import deepcopy
from mne.channels import find_ch_adjacency
from mne.stats import spatio_temporal_cluster_test
import scipy
import pickle

sys.path.append('/sps/cermep/opm/NEW_MEG/mne-python/new_meg')
group_path = '/sps/cermep/opm/NEW_MEG/HV/group/MMN/'

EEG_type=1 #1=SQUIDEEG, 2=OPMEEG, 3=both
load_precomputed=1 #whether to load stats or compute on the fly

import matplotlib.pyplot as plt 
plt.ion()

#Get all files 
if EEG_type==1:
    EEG_evoked = [d for d in listdir(op.join(group_path,'evoked')) if '.fif' in d and 'SQUIDEEG_' in d  and not 'std-' in d]

if EEG_type==2:
    EEG_evoked = [d for d in listdir(op.join(group_path,'evoked')) if '.fif' in d and 'OPMEEG_' in d  and not 'std-' in d]
    
if EEG_type==3:
    EEG_evoked = [d for d in listdir(op.join(group_path,'evoked')) if '.fif' in d and 'EEG_' in d  and not 'std-' in d]    

subs=[f[:2] for f  in EEG_evoked]
subs=np.unique(subs)
evoked=dict()
evoked={s : {'odd':[],'std':[]} for s in subs}

#Load evoked & sort
EEG_evoked.sort()
for f in EEG_evoked:
    if EEG_type<3:    
            sub=f[:2]
            dat=mne.read_evokeds(op.join(group_path,'evoked',f),condition=0)    
            picks=mne.pick_types(dat.info,eeg=True,meg=False,ref_meg=False)
            dat.pick(picks=picks)
            if 'odd' in f:                
                evoked[sub]['odd']=dat
            if 'std' in f:        
                evoked[sub]['std']=dat
    else:
        sub=f[:2]
        dat=mne.read_evokeds(op.join(group_path,'evoked',f),condition=0)    
        picks=mne.pick_types(dat.info,eeg=True,meg=False,ref_meg=False)
        dat.pick(picks=picks)
        if 'odd' in f:                
            evoked[sub]['odd'].append(dat)
        if 'std' in f:        
            evoked[sub]['std'].append(dat)
        
#merge sessions
if EEG_type==3:
    for s in evoked.keys():        
        evoked[s]['odd']=mne.grand_average(evoked[s]['odd'])
        evoked[s]['std']=mne.grand_average(evoked[s]['std'])
    

#create grand average
ev_std_EEG=[]
ev_odd_EEG=[]
for s in evoked.keys():
    ev_std_EEG.append(evoked[s]['std'])    
    ev_odd_EEG.append(evoked[s]['odd'])

#bad channel interpolation
#sub 6 and 13 (OPM) have a missing channel: Fp2
tmp=deepcopy(ev_std_EEG[4])
tmp.data[0,]=ev_std_EEG[5].data[0,]
tmp.data[1,]=np.zeros(np.shape(tmp.data[0,]))
tmp.data[2:,]=ev_std_EEG[5].data[1:,]
tmp.info['bads']=['Fp2']
tmp.interpolate_bads()
ev_std_EEG[5]=tmp

tmp=deepcopy(ev_odd_EEG[4])
tmp.data[0,]=ev_odd_EEG[5].data[0,]
tmp.data[1,]=np.zeros(np.shape(tmp.data[0,]))
tmp.data[2:,]=ev_odd_EEG[5].data[1:,]
tmp.info['bads']=['Fp2']
tmp.interpolate_bads()
ev_odd_EEG[5]=tmp

if EEG_type>1:
    tmp=deepcopy(ev_std_EEG[4])
    tmp.data[0,]=ev_std_EEG[12].data[0,]
    tmp.data[1,]=np.zeros(np.shape(tmp.data[0,]))
    tmp.data[2:,]=ev_std_EEG[12].data[1:,]
    tmp.info['bads']=['Fp2']
    tmp.interpolate_bads()
    ev_std_EEG[12]=tmp

    tmp=deepcopy(ev_odd_EEG[4])
    tmp.data[0,]=ev_odd_EEG[12].data[0,]
    tmp.data[1,]=np.zeros(np.shape(tmp.data[0,]))
    tmp.data[2:,]=ev_odd_EEG[12].data[1:,]
    tmp.info['bads']=['Fp2']
    tmp.interpolate_bads()
    ev_odd_EEG[12]=tmp
    
#grand average
ga_STD_EEG=mne.grand_average(ev_std_EEG)
ga_ODD_EEG=mne.grand_average(ev_odd_EEG)

#add together
evokeds=[]
evokeds.append(ga_STD_EEG)
evokeds[0].comment='std'
evokeds.append(ga_ODD_EEG)
evokeds[1].comment='odd'

mmn=evokeds[0].copy()
mmn.data=evokeds[1].data-evokeds[0].data

#########
# STATS #
#########
EEG_names=['SQUIDEEG','OPMEEG','ALLEEG']

if load_precomputed:
    print('Loading EEG cluster stats')    
    stats_file = open(op.join(group_path, EEG_names[EEG_type-1]+ '_clusterstats.pkl'),'rb')
    cluster_stats=pickle.load(stats_file)
else:

    #get adjacency (neighbors)
    adjacency, ch_names = find_ch_adjacency(ev_std_EEG[0].info, ch_type="eeg")    
    
    #P8-C1 connection seems wrong, remove
    #also FP1-TP9
    tmp=adjacency.toarray()
    P8_ind=[d for d in range(len(ch_names)) if ch_names[d]=='P8']
    C1_ind=[d for d in range(len(ch_names)) if ch_names[d]=='C1']
    tmp[P8_ind[0],C1_ind[0]]=0
    tmp[C1_ind[0],P8_ind[0]]=0
    
    #also Fp1-TP9
    FP1_ind=[d for d in range(len(ch_names)) if ch_names[d]=='Fp1']
    TP9_ind=[d for d in range(len(ch_names)) if ch_names[d]=='TP9']
    if tmp[FP1_ind[0],TP9_ind[0]]:
        tmp[FP1_ind[0],TP9_ind[0]]=0
        tmp[TP9_ind[0],FP1_ind[0]]=0
        
    #remove P7-P8 connection (Across hemisheres)    
    P8_ind=[d for d in range(len(ch_names)) if ch_names[d]=='P8']
    P7_ind=[d for d in range(len(ch_names)) if ch_names[d]=='P7']
    tmp[P8_ind[0],P7_ind[0]]=0
    tmp[P7_ind[0],P8_ind[0]]=0
    
    #FC5-Fp1
    FC5_ind=[d for d in range(len(ch_names)) if ch_names[d]=='FC5']
    FC6_ind=[d for d in range(len(ch_names)) if ch_names[d]=='FC6']
    FP2_ind=[d for d in range(len(ch_names)) if ch_names[d]=='Fp2']
    tmp[FC5_ind[0],FP1_ind[0]]=0
    tmp[FP1_ind[0],FC5_ind[0]]=0
    tmp[FC6_ind[0],FP2_ind[0]]=0
    tmp[FP2_ind[0],FC6_ind[0]]=0
    
    #FC5-TP9 / FC6-TP10
    TP10_ind=[d for d in range(len(ch_names)) if ch_names[d]=='TP10']
    tmp[FC5_ind[0],TP9_ind[0]]=0
    tmp[TP9_ind[0],FC5_ind[0]]=0
    tmp[FC6_ind[0],TP10_ind[0]]=0
    tmp[TP10_ind[0],FC6_ind[0]]=0
    
    #FC5-P7 / FC6-P8
    tmp[FC5_ind[0],P7_ind[0]]=0
    tmp[P7_ind[0],FC5_ind[0]]=0
    tmp[FC6_ind[0],P8_ind[0]]=0
    tmp[P8_ind[0],FC6_ind[0]]=0
    
    #C1-P7 / C2-P8
    C2_ind=[d for d in range(len(ch_names)) if ch_names[d]=='C2']
    tmp[C1_ind[0],P7_ind[0]]=0
    tmp[P7_ind[0],C1_ind[0]]=0
    tmp[C2_ind[0],P8_ind[0]]=0
    tmp[P8_ind[0],C2_ind[0]]=0
    
    
    adjacency2=scipy.sparse.csr_matrix(tmp)
    
    #data
    x1 = [x.get_data() for x in ev_std_EEG]
    x2 = [x.get_data() for x in ev_odd_EEG]
    X=[x1,x2]
    # the dimensions are as expected for the cluster permutation test:
    # n_epochs × n_times × n_channels
    X = [np.transpose(x, (0, 2, 1)) for x in X]
    
    # We are running an F test, so we look at the upper tail
    # see also: https://stats.stackexchange.com/a/73993
    tail = 1
    
    #Critical test statistic
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
        n_permutations=10000,
        threshold=f_thresh,
        tail=tail,
        n_jobs=None,
        buffer_size=None,
        adjacency=adjacency2,
    )
    
    print('Saving stats')       
    stats_path = op.join(group_path, EEG_names[EEG_type-1]+ '_clusterstats.pkl')
    pickle.dump(cluster_stats, open(stats_path, "wb"))
    print('All done.') 
    
F_obs, clusters, p_values, _ = cluster_stats

################
# Plot results #
################

p_accept = 0.05 #corrected p value
good_cluster_inds = np.where(p_values < p_accept)[0]

#plot significant timecourses separately
# loop over clusters
colors = {"odd": "crimson", "std": "steelblue"}    
markers=['o','X','D']
type_name=['SQUIDEEG','OPMEEG','ALLEEG']
topo_av_tw=True #plt topo using an average time window

tws=np.zeros((len(good_cluster_inds),2))
tws_inds=np.zeros((len(good_cluster_inds),2))
if topo_av_tw:
    for i_clu, clu_idx in enumerate(good_cluster_inds):
        
        #unpack cluster information, get unique indices
        time_inds, space_inds = np.squeeze(clusters[clu_idx])
        time_inds = np.unique(time_inds)
        sig_times = ga_STD_EEG.times[time_inds]
        tws[i_clu,] = [sig_times[0], sig_times[-1]]
        tws_inds[i_clu,] = [time_inds[0], time_inds[-1]]
    av_tw=np.average(tws,axis=0)
    av_tw_inds=np.round(np.average(tws_inds,axis=0))

all_masks=[]
best_masks=[]
for i_clu, clu_idx in enumerate(good_cluster_inds):
    
    #unpack cluster information, get unique indices
    time_inds, space_inds = np.squeeze(clusters[clu_idx])
    ch_inds = np.unique(space_inds)
    time_inds = np.unique(time_inds)

    m_dat=mmn.data.T
    f_map = m_dat[time_inds, ...].mean(axis=0)
    
    # create spatial mask
    mask = np.zeros((f_map.shape[0], 1), dtype=bool)
    mask[ch_inds, :] = True
    all_masks.append(mask)

    clust_max=np.where(np.abs(f_map)==np.max(np.abs(f_map[mask[:,0]])))[0][0]
    
    best_mask=np.zeros((f_map.shape[0], 1), dtype=bool)
    best_mask[clust_max,:] = True
    best_masks.append(best_mask)
    
    ch_std=evokeds[0].data[clust_max,]*1e6
    ch_odd=evokeds[1].data[clust_max,]*1e6
    ch_mmn=(ch_odd-ch_std)
    
    # get signals at the sensors contributing to the cluster
    sig_times = ga_STD_EEG.times[time_inds]
    times = ga_STD_EEG.times
        
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))                        
    ax.plot(times,ch_std, label="standard",color=colors['std'])
    ax.plot(times,ch_odd, label="odd",color=colors['odd'])
    ax.plot(times,ch_odd - ch_std, label="MMN",color='k')
    lims=ax.get_xlim()
    ax.hlines(0,lims[0],lims[1],'k')    
    ax.set_xlim(-.1,.4)
    ax.spines.bottom.set_bounds(-.1,.4)    
    lims=ax.get_ylim()            
    ax.vlines(0,lims[0],lims[1],'k',linestyle='--')            
    ax.set_ylim(lims)
    low_bound=ax.get_yticks()[np.where(ax.get_yticks()>ax.get_ylim()[0])[0][0]]
    up_bound=ax.get_yticks()[np.where(ax.get_yticks()<ax.get_ylim()[1])[0][-1]]
    ax.spines.left.set_bounds(low_bound,up_bound)
    ax.set_ylabel("EEG ($\mu$V)")
    ax.set_xlabel("Time (s)")
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    
    ax.fill_betweenx(
        (lims[0], lims[1]), sig_times[0], sig_times[-1], color="orange", alpha=0.3
    )
    
    title = "Cluster #{0}, best sensor: {1}, p={2}".format(i_clu + 1, evokeds[0].ch_names[clust_max],round(p_values[clu_idx],4))
    ax.set_title(title)
    
    ax.legend()
    
    #save
    fig.savefig(op.join(group_path,'Fig2_' +type_name[EEG_type-1]+ '_clust' +str(i_clu+1)+ '_timecourse.pdf'))    
    fig.savefig(op.join(group_path,'Fig2_' +type_name[EEG_type-1]+ '_clust' +str(i_clu+1)+ '_timecourse.png'))    
    
# topoplot
fig, ax = plt.subplots(1, 1, figsize=(8, 4))                        

# plot average test statistic and mark significant sensors
cmap='RdBu_r'
 
f_map = m_dat[(np.arange(int(av_tw_inds[0]),int(av_tw_inds[-1]))), ...].mean(axis=0)

f_evoked = mne.EvokedArray(f_map[:, np.newaxis], ga_STD_EEG.info, tmin=0)
for i,m in enumerate(all_masks):
    f_evoked.plot_topomap(
        times=0,
        mask=m,
        axes=ax,
        cmap=cmap,
        vlim=(np.min, np.max),        
        colorbar=False,
        mask_params=dict(marker=markers[i],markersize=13),
    )    
    f_evoked.plot_topomap(
        times=0,
        mask=best_masks[i],
        axes=ax,
        cmap=cmap,
        vlim=(np.min, np.max),        
        colorbar=False,
        mask_params=dict(marker=markers[i],markersize=25,markerfacecolor='r',markeredgecolor='w')
    )    
        
ax.set_xlabel("Averaged difference map ({:0.3f} - {:0.3f} s)".format(*av_tw[[0, -1]]))

#save
fig.savefig(op.join(group_path,'Fig2_' +type_name[EEG_type-1]+ '_clust_topo.pdf'))    
fig.savefig(op.join(group_path,'Fig2_' +type_name[EEG_type-1]+ '_clust_topo.png'))