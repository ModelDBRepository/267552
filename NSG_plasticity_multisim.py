#!/usr/bin/env python3
"""
Simulate upstate conditions for Patch Samples 4-5 and Matrix Samples 2-3 models.

Modify local channel conductances at site of clustered input for each neuron 
to achieve upstate duration and amplitude consistent with experimental averages.

Do current injection with modified conductances to confirm modifying them does 
not greatly alter the fit to current injection data.

Simulate without blocking sodium channels.

Simulate with additional dispersed inputs.

Simulation steps:

For each neuron:
    - Randomly select parameters from within a range to vary
        - parameters to vary:
        - Random seed necessary for selecting parameters?
    For each set of parameters:
        - Use same random seeds to control synapse selection
        - [done] simulate upstate only: Need an upstate seed (same every sim/param set)
        - [done] Simulate dispersed only: Need a Dispersed seed (same every sim/param set for now)
        - [done] Simulate upstate and dispersed together: Use same upstate seed and same dispersed seed
        - [ ] range over dispersion frequency params
        
        - [later] Should we simulate "spatially disperse" the clustered inputs but at the same time as a control? Not for now
        - [done] Simulate single EPSP (EPSP seed?-same every sim)
        - [done] Simulate IV traces to compare to original model IV to see how much the optimization fit is messed up
        - [ ] simulate upstate plus current injection at increasing steps...

TO DO:

Save voltage at soma from each simulation

File name scheme:
param_set_X_sim_name_sim_variable_name_value_neuron-name_vm.txt
e.g.
param_set_1__upstate_plus_dispersed__dispersed_freq_375__D1PatchSample5_vm.txt



Make plotting optional argument

Save parameter variation values (and any necessary random seeds?) for each simulation

param_set_list.csv

set_ID (corresponds to param_set_X in filenames), var1name, var2name...
e.g.
0,      2, 3,...



"""

import importlib
import numpy as np
from scipy import rand


def moose_main(corticalinput,LTP_amp_thresh_mod=1.158, LTD_amp_thresh_mod=1.656, LTP_dur_thresh_mod=1.653, LTD_dur_thresh_mod=0.867,LTP_gain_mod=0.704, LTD_gain_mod=1.671,nmda_mod=1,seed=42,ClusteringParams=None,randomize=1,global_test=False):
    import logging

    import numpy as np
    np.random.seed(seed)

    from pprint import pprint
    import moose

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
    )
    #from moose_nerp import d1opt as model
    #from moose_nerp import D1MatrixSample2 as model
    from moose_nerp import D1PatchSample5 as model
    from moose_nerp import str_net as net
    from moose_nerp.graph import net_graph, neuron_graph, spine_graph

    # additional, optional parameter overrides specified from with python terminal
    # model.Condset.D1.NaF[model.param_cond.prox] /= 3
    # model.Condset.D1.KaS[model.param_cond.prox] *= 3

    model.param_syn._SynAMPA.Gbar = .25e-9
    model.param_syn._SynNMDA.Gbar = .25e-9
    model.param_syn._SynGaba.Gbar = 1e-9

    net.connect_dict["D1"]["ampa"]["extern1"].dend_loc.postsyn_fraction = 1.0#0.8
    net.param_net.tt_Ctx_SPN.filename = corticalinput
    print(
        "cortical_fraction = {}".format(
            net.connect_dict["D1"]["ampa"]["extern1"].dend_loc.postsyn_fraction
        )
    )
    model.synYN = True
    model.plasYN = True
    model.calYN = True
    model.spineYN = True

    # vary plasticity params
    model.CaPlasticityParams.Plas2_syn.LTP_amp_thresh*=LTP_amp_thresh_mod
    model.CaPlasticityParams.Plas2_syn.LTD_amp_thresh*=LTD_amp_thresh_mod
    model.CaPlasticityParams.Plas2_syn.LTP_dur_thresh*=LTP_dur_thresh_mod
    model.CaPlasticityParams.Plas2_syn.LTD_dur_thresh*=LTD_dur_thresh_mod
    model.CaPlasticityParams.Plas2_syn.LTP_gain*=LTP_gain_mod
    model.CaPlasticityParams.Plas2_syn.LTD_gain*=LTD_gain_mod
    
    # new changes: Universal function to modify variables. Needs to extract any attribute based (e.g. model.CaParams...) or dict/list based (e.g. model.dict['key']) parameter name as a string, parse the variable, and apply the modification. Additionally, need to specify a multiplicative change, additive change, or direct replacement.
    #model.CaPlasticityParams.BufferCapacityDensity*=buffer_capacity_density_gain_mod
    
    ############################# 
    ## Set spatial clustering params for clustering spines ##
    if ClusteringParams is not None:
        model.SpineParams.ClusteringParams = {}
        model.SpineParams.ClusteringParams['n_clusters'] = ClusteringParams['n_clusters']
        model.SpineParams.ClusteringParams['cluster_length'] = ClusteringParams['cluster_length']
        model.SpineParams.ClusteringParams['n_spines_per_cluster'] = ClusteringParams['n_spines_per_cluster']

    net.single = True
    model.ConcOut = model.param_cond.ConcOut = 1.2
    create_model_sim.setupOptions(model)
    param_sim = model.param_sim
    param_sim.useStreamer = True
    param_sim.plotdt = 0.1e-3
    param_sim.stim_loc = model.NAME_SOMA
    param_sim.stim_paradigm = "inject"
    param_sim.injection_current = [0]  # [-0.2e-9, 0.26e-9]
    param_sim.injection_delay = 0.2
    param_sim.injection_width = 0.4
    param_sim.simtime = 2 if not global_test else .01#21.5#3.5  # 21
    net.num_inject = 0
    model.name = model.__name__.split('.')[-1]
    net.confile = "str_connect_plas_sim{}_{}_corticalfraction_{}".format(
        model.name, net.param_net.tt_Ctx_SPN.filename, 0.8
    )

    print('randomize',randomize,'global test',global_test,'simtime',param_sim.simtime) #whether or not to randomize cortical trains
    if net.num_inject == 0:
        param_sim.injection_current = [0]
    #################################-----------create the model: neurons, and synaptic inputs
    model = create_model_sim.setupNeurons(model, network=not net.single)
    all_neur_types = model.neurons
    # FSIsyn,neuron = cell_proto.neuronclasses(FSI)
    # all_neur_types.update(neuron)

    ## Limit input trains to 2 seconds and shuffle ISIs if ran=1:
    ttables.TableSet.create_all()
    randomize_input_trains(net.param_net.tt_Ctx_SPN,ran=randomize)

    population, connections, plas = create_network.create_network(
        model, net, all_neur_types,create_all=False
    )
    
    ###### Set up stimulation - could be current injection or plasticity protocol
    # set num_inject=0 to avoid current injection
    if net.num_inject < np.inf:
        inject_pop = inject_func.inject_pop(population["pop"], net.num_inject)
    else:
        inject_pop = population["pop"]
    # Does setupStim work for network?
    # create_model_sim.setupStim(model)
    pg = inject_func.setupinj(
        model, param_sim.injection_delay, param_sim.injection_width, inject_pop
    )
    moose.showmsg(pg)

    ##############--------------output elements
    if net.single:
        # fname=model.param_stim.Stimulation.Paradigm.name+'_'+model.param_stim.location.stim_dendrites[0]+'.npz'
        # simpath used to set-up simulation dt and hsolver
        simpath = ["/" + neurotype for neurotype in all_neur_types]
        create_model_sim.setupOutput(model)
    else:  # population of neurons
        spiketab, vmtab, plastab, catab = net_output.SpikeTables(
            model, population["pop"], net.plot_netvm, plas, net.plots_per_neur
        )
        # simpath used to set-up simulation dt and hsolver
        simpath = [net.netname]
        clocks.assign_clocks(
            simpath,
            param_sim.simdt,
            param_sim.plotdt,
            param_sim.hsolve,
            model.param_cond.NAME_SOMA,
        )
    if model.synYN and (param_sim.plot_synapse or net.single):
        # overwrite plastab above, since it is empty
        syntab, plastab, stp_tab = tables.syn_plastabs(connections, model)
        nonstim_plastab = tables.nonstimplastabs(plas)

    # Streamer to prevent Tables filling up memory on disk
    # This is a hack, should be better implemented
    if param_sim.useStreamer == True:
        allTables = moose.wildcardFind("/##[ISA=Table]")
        streamer = moose.Streamer("/streamer")
        streamer.outfile = "testdata/plas_sim{}_{}_glu{}_ran{}_seed_{}.npy".format(model.name, net.param_net.tt_Ctx_SPN.filename, 
                                                                                   str(randomize),str(model.SYNAPSE_TYPES['ampa'].spinic),seed)
        moose.setClock(streamer.tick, 0.1)
        for t in allTables:
            if any(s in t.path for s in ["plas", "VmD1_0", "extern", "Shell_0"]):
                streamer.addTable(t)
            else:
                t.tick = -2

    ################### Actually run the simulation
    def run_simulation(injection_current, simtime):
        print(u"◢◤◢◤◢◤◢◤ injection_current = {} ◢◤◢◤◢◤◢◤".format(injection_current))
        pg.firstLevel = injection_current
        moose.reinit()
        moose.start(simtime, True)

    import os
    print('does outfile {} exist before sim: {}'.format(streamer.outfile,os.path.exists(streamer.outfile)))
    traces, names = [], []
    for inj in param_sim.injection_current:
        run_simulation(injection_current=inj, simtime=param_sim.simtime)
    print('does outfile {} exist AFTER sim: {}'.format(streamer.outfile,os.path.exists(streamer.outfile)))
    #weights = [w.value for w in moose.wildcardFind("/##/plas##[TYPE=Function]")]
    #import matplotlib.pyplot as plt
    #plt.figure()
    #plt.hist(weights, bins=100)
    #plt.title("plas_sim_{}".format(net.param_net.tt_Ctx_SPN.filename))
    #plt.savefig("plas_simd1opt_{}.png".format(net.param_net.tt_Ctx_SPN.filename))
    if param_sim.useStreamer == True:
        import atexit

        atexit.register(moose.quit)
    #return weights

def randomize_input_trains(timetable,ran=1, maxtime=2):
    for tt in timetable.stimtab:
        v = tt[0].vector # copies time table vector to v (doesn't reference it)
        v = v[v<=maxtime] # Use only first trial, i.e. spike times less than 2
        if ran:
            diff = np.diff(v,prepend=0) # compute ISIs using diff
            np.random.shuffle(diff) # Shuffle ISIs to randomize timing pattern
            v = np.cumsum(diff) # create new spike train from shuffle differences
        tt[0].vector = v # Replace timetable vector with new values
    ##import pdb;pdb.set_trace()
    return timetable



def subprocess_main(function, corticalinput,kwds,time_limit):
    print('enter subprocess_main')
    from multiprocessing import Process, Queue
    import time
    # q = Queue()
    p = Process(target=function, args=(corticalinput,), kwargs=kwds)
    p.start()
    
    # result = q.get()
    # print(result)
    remaining = time_limit - time.time()
    if remaining <=0:
        p.terminate()
        return
    p.join(timeout=remaining-10)
    p.terminate()

    # return result

def make_rand_mod_dict(n=100):
    #plas_mod_values =[.5,2]
    #plas_mod_keys = ['LTP_amp_thresh_mod', 'LTD_amp_thresh_mod','LTP_dur_thresh_mod', #'LTD_dur_thresh_mod','LTP_gain_mod', 'LTD_gain_mod']
    #rand_mod_list = np.random.uniform(*plas_mod_values,size=(n,6))
    #rand_mod_dicts = [{k:v for k,v in zip(plas_mod_keys,r)} for r in rand_mod_list]
    np.random.seed(42)
    #ClusteringParams = {'n_clusters':20, 'cluster_length':30e-6, 'n_spines_per_cluster':10}
    rand_mod_dicts = [] # List of dictionaries, each element of list is a dictionary contianing set of parameter modification values that will be passed to main moose function.
    for i in range(n):
        mod_dict = {}
        mod_dict['seed']=np.random.randint(100000)
        mod_dict['ClusteringParams'] = {}
        # vary number of clutsters between 1 and 30
        n_clusters = np.random.randint(1,30)
        # Need to relatively fix total number of spines and cluster length relative to n_clusters
        total_spines = 280#200 # Minimum -- note that we round up n spines per cluster, should probably be a parameter
        n_spines_per_cluster = int(np.ceil(total_spines/n_clusters))
        mean_cluster_length = 200e-6/n_clusters # 200 microns divided by number of clusters, so if we have fewer clusters we have a greater cluster length to distribute within. This value might need to be tweaked
        cluster_length = np.random.normal(mean_cluster_length, 0.1*mean_cluster_length) # randomize cluster length somewhat
        mod_dict['ClusteringParams']['n_clusters'] = n_clusters
        mod_dict["ClusteringParams"]['cluster_length']=cluster_length
        mod_dict['ClusteringParams']['n_spines_per_cluster']=n_spines_per_cluster
        rand_mod_dicts.append(mod_dict)
    return rand_mod_dicts


def mpi_main(num_sims=100,randomize=1,global_test=False):
    if __name__ == "__main__":
        print('running mpi main')
        from mpi4py import MPI
        from mpi4py.futures import MPICommExecutor
        import time
        import pickle

        with MPICommExecutor(MPI.COMM_WORLD, root=0) as executor:
            time_limit = time.time() + 60 * 60 * 8 if not global_test else time.time() + 60*60*.1#3.75  # 3.75 hours
            
            if executor is not None:
                print(executor)
                results = []
                make_new_params = True
                if make_new_params:
                    n = num_sims if not global_test else 52
                    param_set_list = make_rand_mod_dict(n=n)
                    with open("testparams.pickle", "wb") as f:
                        pickle.dump(param_set_list, f)
                else:
                    with open("testparams.pickle",'rb') as f:
                        param_set_list = pickle.load(f)
                print(param_set_list)
                for i, param_set in enumerate(param_set_list):#1020
                    param_set['randomize']=randomize
                    print(i, param_set)
                    r = executor.submit(
                                subprocess_main, *(moose_main,"FullTrialLowVariabilitySimilarTrialsTruncatedNormal", param_set,  time_limit)
                            )
                    results.append(r)

                while True:
                    if all([res.done() for res in results]):
                        print('all results returned done; breaking')
                        #import pdb;pdb.set_trace()
                        break

                    if time.time() >= time_limit:
                        print("****************** TIME LIMIT EXCEEDED***********")
                        for res in results:
                            res.cancel()
                            #print('canceling', res)
                            
                        #executor.shutdown(wait=False)
                        print('shutting down')
                        MPI.COMM_WORLD.Abort()
                        print('aborting')
                        break
                print('done')
                return
            #while True:
            #    if time.time() >= time_limit:
            #        break
            #MPI.COMM_WORLD.Abort()


### Notes
### What to randomize on each simulation:
### 1. Spatiotemporal input pattern. Select X+/-i in vivo spike trains, jitter spike times by Y, move spikes to different trains with prob Z, shuffle ISIs?
### 2. Neuron type: patch or matrix?
### 3. Soma NaF: blocked or not?
### 4. Other model parameters: varied +/- 10%? Which parameters?
'''
if __name__ == "__main__":

    import sys

    args = sys.argv
    # args.append('--single')
    if "--test" in args:
        global_test = True
    print('global test = {}'.format(global_test))
    if len(args) > 1 and args[1] == "--single":
        # upstate_main(list(mod_dict.keys())[0],mod_dict)
        ClusteringParams = {'n_clusters':20, 'cluster_length':20e-6, 'n_spines_per_cluster':10}
        moose_main("FullTrialLowVariabilitySimilarTrialsTruncatedNormal", seed=42,ClusteringParams=ClusteringParams)

    elif len(args) > 1 and args[1] == "--iv":
        # upstate_main(list(mod_dict.keys())[0],mod_dict)
        iv_main("D1PatchSample5", mod_dict, filename="test")

    elif len(args) > 1 and args[1] == "--mp":
        results = []
        from multiprocessing import Pool

        with Pool(16, maxtasksperchild=1) as p:
            param_set_list = [rand_mod_dict() for i in range(10000)]

            import pickle

            with open("params.pickle", "wb") as f:
                pickle.dump(param_set_list, f)

            #print(param_set_list)
            for i, param_set in enumerate(param_set_list):
                for key in mod_dict:
                    for sim in sims:
                        # param_set_1__upstate_plus_dispersed__dispersed_freq_375__D1PatchSample5_vm.txt

                        filename = (
                            "param_set_"
                            + str(i)
                            + "__"
                            + sim["name"]
                            + "__dispersed_freq_"
                            + str(sim["kwds"].get("freq_dispersed"))
                            + "__"
                            + key
                        )
                        kwds = {k: v for k, v in sim["kwds"].items()}
                        kwds["filename"] = filename
                        # r = p.apply_async(upstate_main, args=(key, mod_dict),kwds={'num_dispersed':0})
                        r = p.apply_async(sim["f"], args=(key, param_set), kwds=kwds)
                        results.append(r)
            for res in results:
                res.wait()
    else:  #when running on NSG
        mpi_main()
        print('done?')
'''
