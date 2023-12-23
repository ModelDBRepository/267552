# -*- coding:utf-8 -*-

######## SPnetSim.py ############
"""\
Create a network of SP neurons using dictionaries for channels, synapses, and network parameters

Can use ghk for calcium permeable channels if ghkYesNo=1
Optional calcium concentration in compartments (calcium=1)
Optional synaptic plasticity based on calcium (plasyesno=1)
Spines are optional (spineYesNo=1), but not allowed for network
The graphs won't work for multiple spines per compartment
"""
from __future__ import print_function, division
import logging

import numpy as np
np.random.seed(2020)
import matplotlib.pyplot as plt
plt.ion()

from pprint import pprint
import moose

from moose_nerp.prototypes import (create_model_sim,
                                   cell_proto,
                                   clocks,
                                   inject_func,
                                   create_network,
                                   tables,
                                   net_output,
                                   logutil,
                                   util,
                                   standard_options)
#from moose_nerp import d1opt as model
from moose_nerp import D1MatrixSample2 as model
from moose_nerp import str_net as net
from moose_nerp.graph import net_graph, neuron_graph, spine_graph

#additional, optional parameter overrides specified from with python terminal
#model.Condset.D1.NaF[model.param_cond.prox] /= 3
#model.Condset.D1.KaS[model.param_cond.prox] *= 3
net.connect_dict['D1']['ampa']['extern1'].dend_loc.postsyn_fraction = 0.7#0.7
net.param_net.tt_Ctx_SPN.filename = 'FullTrialLowVariabilitySimilarTrials'
#net.param_net.tt_Ctx_SPN.filename = 'FullTrialLowVariability'
#net.param_net.tt_Ctx_SPN.syn_per_tt = 2
model.synYN = True
model.plasYN = True
model.calYN = True
model.spineYN = True
net.single=True
#for k,v in model.param_ca_plas.CaShellModeDensity.items():
#    model.param_ca_plas.CaShellModeDensity[k] = model.param_ca_plas.SHELL
create_model_sim.setupOptions(model)
param_sim = model.param_sim
param_sim.useStreamer = True
param_sim.plotdt = .1e-3
param_sim.stim_loc = model.NAME_SOMA
param_sim.stim_paradigm = 'inject'
param_sim.injection_current = [0] #[-0.2e-9, 0.26e-9]
param_sim.injection_delay = 0.2
param_sim.injection_width = 0.4
param_sim.simtime = 1.1#.424#5#.42408#.42409#.01#.7#3#.#21
#param_sim.simdt=.5e-05
# Notes: at .42409, Calcium concentration goes negative in some shells, for example, in  /D1[0]/secdend42[0]/sp0head[0]/Shell_0[0]. Also,at this time step, the ICa of the nmda channels goes to NaN.moose Then, at .4241, Concentration becomes NaN, then, at the next time step, voltages go to nan. Why this doesn't happen without the BKCa channel is unclear...

## TODO: Lower calcium to 1.5 or 1.2; thought to be lower in vivo than the 2.0 used commonly in brain slice experiments




#param_sim.plot_channels = "BKCa"
#param_sim.plotgate='BKCa'

net.num_inject = 0
if net.num_inject==0:
    param_sim.injection_current=[0]
#################################-----------create the model: neurons, and synaptic inputs
model=create_model_sim.setupNeurons(model,network=not net.single)
all_neur_types=model.neurons
#FSIsyn,neuron = cell_proto.neuronclasses(FSI)
#all_neur_types.update(neuron)
population,connections,plas=create_network.create_network(model, net, all_neur_types)

###### Set up stimulation - could be current injection or plasticity protocol
# set num_inject=0 to avoid current injection
if net.num_inject<np.inf :
    inject_pop=inject_func.inject_pop(population['pop'],net.num_inject)
else:
    inject_pop=population['pop']
#Does setupStim work for network?
#create_model_sim.setupStim(model)
pg=inject_func.setupinj(model, param_sim.injection_delay,param_sim.injection_width,inject_pop)
moose.showmsg(pg)

##############--------------output elements
if net.single:
    #fname=model.param_stim.Stimulation.Paradigm.name+'_'+model.param_stim.location.stim_dendrites[0]+'.npz'
    #simpath used to set-up simulation dt and hsolver
    simpath=['/'+neurotype for neurotype in all_neur_types]
    create_model_sim.setupOutput(model)
else:   #population of neurons
    spiketab,vmtab,plastab,catab=net_output.SpikeTables(model, population['pop'], net.plot_netvm, plas, net.plots_per_neur)
    #simpath used to set-up simulation dt and hsolver
    simpath=[net.netname]
    clocks.assign_clocks(simpath, param_sim.simdt, param_sim.plotdt, param_sim.hsolve,model.param_cond.NAME_SOMA)
if model.synYN and (param_sim.plot_synapse or net.single):
    #overwrite plastab above, since it is empty
    syntab, plastab, stp_tab=tables.syn_plastabs(connections,model)
    nonstim_plastab = tables.nonstimplastabs(plas)


# Streamer to prevent Tables filling up memory on disk
# This is a hack, should be better implemented
if param_sim.useStreamer==True:
    allTables = moose.wildcardFind('/##[ISA=Table]')
    streamer = moose.Streamer('/streamer')
    streamer.outfile = 'plas_sim_{}.npy'.format(net.param_net.tt_Ctx_SPN.filename)
    moose.setClock(streamer.tick,0.1)
    for t in allTables:
        if any (s in t.path for s in ['plas','VmD1_0','extern','Shell_0']):
            streamer.addTable(t)
        else:
            t.tick=-2

spinedistdict = {}
for sp in moose.wildcardFind('D1/##/#head#[ISA=CompartmentBase]'):
    dist,_ = util.get_dist_name(sp)
    path = sp.path
    spinedistdict[path]=dist
    
################### Actually run the simulation
def run_simulation(injection_current, simtime, continue_sim = False):
    print(u'◢◤◢◤◢◤◢◤ injection_current = {} ◢◤◢◤◢◤◢◤'.format(injection_current))
    pg.firstLevel = injection_current
    if not continue_sim:
        moose.reinit()
    moose.start(simtime)

# NMDA Channel reverses? Store values in table
# nmda_ICa_table = moose.Table('/data/secdend42_sp0head_iCa')
# nmda_chan = moose.element('/D1/secdend42/sp0head/nmda')
# moose.connect(nmda_ICa_table,'requestOut',nmda_chan,'getICa')

    
continue_sim = False
traces, names = [], []
for inj in param_sim.injection_current:
    run_simulation(injection_current=inj, simtime=param_sim.simtime,continue_sim=continue_sim)

    # Debug simulation steps
    while True:
        compvms = np.array([comp.Vm for comp in moose.wildcardFind('/D1/##[ISA=ZombieCompartment]')])
        compvmnans = np.argwhere(np.isnan(compvms))
        compshells = np.array([comp.C for comp in moose.wildcardFind('/D1/##[ISA=DifShell]')]   )
        for comp in moose.wildcardFind('/D1/##[ISA=DifShell]'):
            if np.isnan(comp.C):
                moose.showfields(comp)
        if len(compvmnans > 0):
            break
        if moose.element('/clock').currentTime > 1:
            break
        moose.showfields('/Clock')
        print('continuing simulation')
        moose.start(1e-05)
    import IPython
    IPython.embed()
    if getattr(model.param_sim,'plotgate',None):
        plt.figure()
        ts = np.linspace(0, model.param_sim.simtime, len(model.gatetables['gatextab'].vector))
        plt.suptitle('X,Y,Z gates; hsolve='+str(model.param_sim.hsolve)+' calYN='+str(model.calYN)+' Zgate='+str(model.Channels[model.param_sim.plotgate][0][2]))
        plt.plot(ts,model.gatetables['gatextab'].vector,label='X')
        plt.plot(ts,model.gatetables['gateytab'].vector,label='Y')
        if model.Channels[model.param_sim.plotgate][0][2]==1:
            plt.plot(ts,model.gatetables['gateztab'].vector,label='Z')
        plt.legend()

    if net.single and len(model.vmtab):
        for neurnum,neurtype in enumerate(util.neurontypes(model.param_cond)):
            traces.append(model.vmtab[neurtype][0].vector)
            names.append('{} @ {}'.format(neurtype, inj))
        if model.synYN:
            net_graph.syn_graph(connections, syntab, param_sim)
        if model.plasYN:
            net_graph.syn_graph(connections, plastab, param_sim, graph_title='Plas Weight')
            net_graph.syn_graph(connections, nonstim_plastab, param_sim, graph_title='NonStim Plas Weight')

        if model.spineYN:
            spine_graph.spineFig(model,model.spinecatab,model.spinevmtab, param_sim.simtime)
    else:
        if net.plot_netvm:
            net_graph.graphs(population['pop'], param_sim.simtime, vmtab,catab,plastab)
        if model.synYN and param_sim.plot_synapse:
            net_graph.syn_graph(connections, syntab, param_sim)
        net_output.writeOutput(model, net.outfile+str(inj),spiketab,vmtab,population)

if net.single:
    neuron_graph.SingleGraphSet(traces, names, param_sim.simtime)
    # block in non-interactive mode

weights = [w.value for w in moose.wildcardFind('/##/plas##[TYPE=Function]')]
#import IPython
#IPython.embed()

plt.figure()
plt.hist(weights,bins=100)
util.block_if_noninteractive()

if param_sim.useStreamer==True:
    import atexit
    atexit.register(moose.quit)

'''
import detect
if net.single:
    vmtab=model.vmtab
spike_time={key:[] for key in population['pop'].keys()}
numspikes={key:[] for key in population['pop'].keys()}
for neurtype, tabset in vmtab.items():
    for tab in tabset:
       spike_time[neurtype].append(detect.detect_peaks(tab.vector)*param_sim.plotdt)
    numspikes[neurtype]=[len(st) for st in spike_time[neurtype]]
    print(neurtype,'mean:',np.mean(numspikes[neurtype]),'rate',np.mean(numspikes[neurtype])/param_sim.simtime,'from',numspikes[neurtype], 'spikes')
#spikes=[st.vector for tabset in spiketab for st in tabset]
'''
