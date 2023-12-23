#%%
#### This file can be used to create the spine to spine distance file
import numpy as np
import moose
from moose_nerp import D1PatchSample5 as model
from moose_nerp.prototypes import (
        create_model_sim,
        cell_proto,
        clocks,
        inject_func,
        create_network,
        tables,
        net_output,
        logutil,

        util,
        standard_options,
        
        ttables,
        spines,
    
    )
from moose_nerp import str_net as net
from NSG_plasticity_multisim import randomize_input_trains

seed=42
test_connect=False
np.random.seed(seed)

##### Parameters changed in main
model.synYN = True
model.plasYN = True
model.calYN = True
model.spineYN = True
net.single = True
net.connect_dict["D1"]["ampa"]["extern1"].dend_loc.postsyn_fraction = 1.0#0.8
net.param_net.tt_Ctx_SPN.filename = "FullTrialLowVariabilitySimilarTrialsTruncatedNormal"
randomize=1 #whether or not to randomize cortical trains

total_spines = 280 # Minimum -- note that we round up n spines per cluster
####### Note: if synapses are created on dendrites, then even with only 200 spines, a mean of 281 +/- 36 synapses are created (23 trials)
####### with 200 spines and postsyn_fraction = 0.8, 160 ampa synapses were created
####### with 250 spines and postsyn_fraction = 0.8, 208 ampa synapses were created
####### with 250 spines and postsyn_fraction = 1.0, 260 ampa synapses were created
####### with 280 spines and postsyn_fraction = 1.0, 280 ampa synapses were created

model.SpineParams.ClusteringParams = {}
model.SpineParams.ClusteringParams['n_clusters'] = 20
model.SpineParams.ClusteringParams['cluster_length'] = 200e-6/model.SpineParams.ClusteringParams['n_clusters']# 200 microns divided by number of clusters, fewer clusters --> greater cluster length to distribute within.
model.SpineParams.ClusteringParams['n_spines_per_cluster'] = int(np.ceil(total_spines/model.SpineParams.ClusteringParams['n_clusters']))

######## create model
create_model_sim.setupOptions(model)
model.name = model.__name__.split('.')[-1]
model = create_model_sim.setupNeurons(model, network=not net.single)
all_neur_types = model.neurons#util.neurontypes(model.param_cond)

if test_connect:
    ttables.TableSet.create_all()
    randomize_input_trains(net.param_net.tt_Ctx_SPN,ran=randomize)

    population, connections, plas = create_network.create_network(
        model, net, all_neur_types,create_all=False)

    outfile="testdata/plas_sim{}_{}_glu{}_ran{}_seed_{}.npy".format(model.name, net.param_net.tt_Ctx_SPN.filename, 
                                                                str(randomize),str(model.SYNAPSE_TYPES['ampa'].spinic),seed)
    print('test finished, outfile name=',outfile)
#%%
'''
#this is done in spines.py
possible_spines = spines.getPossibleSpines(model,ntypes[0],model.ghkYN,model.NAME_SOMA)
spine_to_spine_dist_array = spines.possible_spine_to_spine_distances(model,possible_spines)
chosen_spine_clusters,chosen_spines_in_each_cluster,all_chosen_spines = spines.choose_spine_clusters(model,possible_spines,spine_to_spine_dist_array,20,20e-6,10)
np.save('s2sdist.npy', spine_to_spine_dist_array)
np.save('clusters.npz',clusters=chosen_spine_clusters,cluster_spines=chosen_spines_in_each_cluster,all_spines=all_chosen_spines)
'''
# %%
'''
from moose_nerp.prototypes import spatiotemporalInputMapping as stim
neuron = list(model.neurons.values())[0][0]
bd = stim.getBranchDict(neuron)
comp_to_branch_dict = stim.mapCompartmentToBranch(neuron)
'''
'''
def compute_spine_to_spine_dist(spine, other_spine,print_info=False):
    #Compute the path distance along dendritic tree between any two spines
    # get parent compartment of spine
    spine_parents = [spine['parentComp'], other_spine['parentComp']]

    # get the branch of the spine_parent
    spine_branches = [comp_to_branch_dict[sp.path] for sp in spine_parents]
    branch_paths = spine_branches[0]['BranchPath'], spine_branches[1]['BranchPath']
    # if on same branch
    if spine_branches[0]==spine_branches[1]:
        # if on same compartment:
        if spine_parents[0]==spine_parents[1]:
            spine_to_spine_dist = np.sqrt((spine['x'] - other_spine['x'])**2 + (spine['y'] - other_spine['y'])**2 + (spine['z'] - other_spine['z'])**2)
        # else if on same branch but not same compartment:
        else:
            compdistances = [bd[sb['Branch']]['CompDistances'] for sb in spine_branches]
            complists = [bd[sb['Branch']]['CompList'] for sb in spine_branches]
            compindexes = [cl.index(spine_parent.path) for cl,spine_parent in zip(complists, spine_parents)]
            comp_to_comp_distance = np.abs(compdistances[0][compindexes[0]] - compdistances[1][compindexes[1]])
            spine_to_spine_dist = comp_to_comp_distance
    # else if on different branches, find common parent branch first
    else:
        for a,b in zip(branch_paths[0], branch_paths[1]):
            #print(a,b)
            if a == b:
                common_parent = a
        common_parent_distance = bd[common_parent]['CompDistances'][0]
        if print_info:
            print('common parent is ', common_parent, 'common_parent_distance is ', common_parent_distance)
        allcompdistances = [bd[sb['Branch']]['CompDistances'] for sb in spine_branches]
        complists = [bd[sb['Branch']]['CompList'] for sb in spine_branches]
        compindexes = [cl.index(spine_parent.path) for cl,spine_parent in zip(complists, spine_parents)]
        compdistances = [allcompdistances[0][compindexes[0]], allcompdistances[1][compindexes[1]]]
        comp_to_comp_distance = (compdistances[0]-common_parent_distance) + (compdistances[1]-common_parent_distance)
        if print_info:
            print('compdistances',compdistances,'comp_to_comp_distance', comp_to_comp_distance)
        spine_to_spine_dist = comp_to_comp_distance
    return spine_to_spine_dist
'''
'''
s2s = compute_spine_to_spine_dist(possible_spines[0],possible_spines[1])
# #%%
spine_to_spine_dist_array = np.zeros((len(possible_spines), len(possible_spines)))
import itertools
for spine_pairs in itertools.combinations(range(len(possible_spines)), 2):
    spine_dist = compute_spine_to_spine_dist(possible_spines[spine_pairs[0]],possible_spines[spine_pairs[1]])
    spine_to_spine_dist_array[spine_pairs[0], spine_pairs[1]] = spine_dist
    spine_to_spine_dist_array[spine_pairs[1], spine_pairs[0]] = spine_dist
'''
# # %%
