# Figure 2 - SQUID results, permutation cluster stats

#Preamble
import os.path as op
import sys
import mne
import numpy as np
from os import listdir
from mne.channels import find_ch_adjacency
from mne.stats import spatio_temporal_cluster_test
import scipy
import pickle

sys.path.append('/sps/cermep/opm/NEW_MEG/mne-python/new_meg')
group_path = '/sps/cermep/opm/NEW_MEG/HV/group/MMN/'

import matplotlib.pyplot as plt 
plt.ion()

load_precomputed=1 #whether to load stats or compute on the fly

#Get all files 
SQUID_evoked = [d for d in listdir(op.join(group_path,'evoked')) if '.fif' in d and 'SQUID_' in d  and not 'std-' in d]

#get subs
subs=[f[:2] for f  in SQUID_evoked]
subs=np.unique(subs)

#Load evoked & sort
evoked=dict()
evoked={s : {'odd':[],'std':[]} for s in subs}
SQUID_evoked.sort()
for f in SQUID_evoked:
    sub=f[:2]
    dat=mne.read_evokeds(op.join(group_path,'evoked',f),condition=0)    
    if 'odd' in f:                
        evoked[sub]['odd']=dat
    if 'std' in f:        
        evoked[sub]['std']=dat
    

#create grand average
ev_std_SQUID=[]
ev_odd_SQUID=[]
for s in evoked.keys():
    picks=mne.pick_types(evoked[s]['std'].info,meg=True,ref_meg=False)
    ev_std_SQUID.append(evoked[s]['std'].pick(picks=picks))
    picks=mne.pick_types(evoked[s]['odd'].info,meg=True,ref_meg=False)
    ev_odd_SQUID.append(evoked[s]['odd'].pick(picks=picks))

ga_STD_SQUID=mne.grand_average(ev_std_SQUID)
ga_ODD_SQUID=mne.grand_average(ev_odd_SQUID)

#add together
evokeds=[]
evokeds.append(ga_STD_SQUID)
evokeds[0].comment='std'
evokeds.append(ga_ODD_SQUID)
evokeds[1].comment='odd'

mmn=evokeds[0].copy()
mmn.data=evokeds[1].data-evokeds[0].data

#########
# STATS #
#########

if load_precomputed:
    print('Loading SQUID cluster stats')
    stats_file = open(op.join(group_path, "SQUID_clusterstats.pkl"),'rb')
    cluster_stats=pickle.load(stats_file)
else:
    
    #get adjacency (neighbors)
    adjacency, ch_names = find_ch_adjacency(ga_STD_SQUID.info, ch_type="mag")
    
    #MRT36 gets added to the adjacency matrix, but does not exist in our data
    mis_ind=np.where([1 if c=='MRT36' else 0 for c in ch_names])[0][0]
    
    #remove from adjacency matrix
    tmp=adjacency.toarray()
    tmp=np.delete(tmp,mis_ind,axis=1)
    tmp=np.delete(tmp,mis_ind,axis=0)
    adjacency2=scipy.sparse.csr_matrix(tmp)
    
    ch_names2=[c+'-2805' for c in ch_names if c!='MRT36']
    
    #prep data
    x1 = [x.get_data() for x in ev_std_SQUID]
    x2 = [x.get_data() for x in ev_odd_SQUID]
    X=[x1,x2]
    
    # n_epochs × n_times × n_channels
    X = [np.transpose(x, (0, 2, 1)) for x in X]
    
    # We are running an F test, so we look at the upper tail
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
        n_permutations=10000,
        threshold=f_thresh,
        tail=tail,
        n_jobs=None,
        buffer_size=None,
        adjacency=adjacency2,
    )
    
    print('Saving stats')       
    stats_path = op.join(group_path, "SQUID_clusterstats.pkl")
    pickle.dump(cluster_stats, open(stats_path, "wb"))
    print('All done.') 

F_obs, clusters, p_values, _ = cluster_stats

################
# Plot results #
################

#plot significant timecourses separately
p_accept = 0.05 #corrected p-value
good_cluster_inds = np.where(p_values < p_accept)[0]

#plot params
colors = {"odd": "crimson", "std": "steelblue"}    
markers=['o','X','D']

#loop over clusters
all_time_inds=[]
all_masks=[]
best_masks=[]
for i_clu, clu_idx in enumerate(good_cluster_inds):
    
    #unpack cluster information, get unique indices
    time_inds, space_inds = np.squeeze(clusters[clu_idx])
    ch_inds = np.unique(space_inds)
    time_inds = np.unique(time_inds)
    all_time_inds.append(time_inds)

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
    
    ch_std=evokeds[0].data[clust_max,]*1e15
    ch_odd=evokeds[1].data[clust_max,]*1e15
    ch_mmn=(ch_odd-ch_std)
    
    # get signals at the sensors contributing to the cluster
    sig_times = ga_STD_SQUID.times[time_inds]
    times = ga_STD_SQUID.times
        
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
    ax.set_ylabel("MEG (fT)")
    ax.set_xlabel("Time (s)")
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
            
    ax.fill_betweenx(
        (lims[0], lims[1]), sig_times[0], sig_times[-1], color="orange", alpha=0.3
    )

    title = "Cluster #{0}, best sensor: {1}, p={2}".format(i_clu + 1, evokeds[0].ch_names[clust_max][:-5],round(p_values[clu_idx],4))
    ax.set_title(title)

    ax.legend()
    
    #save
    fig.savefig(op.join(group_path,'Fig2_SQUID_clust' +str(i_clu+1)+ '_timecourseOnly.pdf'))    
    fig.savefig(op.join(group_path,'Fig2_SQUID_clust' +str(i_clu+1)+ '_timecourseOnly.png'))    
    
    
#Plot topo, using average significant window    
# initialize figure
fig, ax = plt.subplots(1, 1, figsize=(8, 4))                        

#Get average significant time window
start=int(np.mean([all_time_inds[0][0],all_time_inds[1][0]]))
stop=int(np.mean([all_time_inds[0][-1],all_time_inds[1][-1]]))
f_map = m_dat[np.arange(start,stop,1), ...].mean(axis=0)
sig_times = ga_STD_SQUID.times[time_inds]

# plot average test statistic and mark significant sensors
cmap='RdBu_r'

f_evoked = mne.EvokedArray(f_map[:, np.newaxis], ga_STD_SQUID.info, tmin=0)
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
ax.set_xlabel("Averaged difference map ({:0.3f} - {:0.3f} s)".format(*sig_times[[0, -1]]))

#save
fig.savefig(op.join(group_path,'Fig2_SQUID_topo.pdf'))    
fig.savefig(op.join(group_path,'Fig2_SQUID_topo.png'))    