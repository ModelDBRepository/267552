# Additional iv traces for example figures for Plotkin collaboration
import pickle
import importlib
import numpy as np

#import sim_upstate

# load param list
with open("/home/dbd/NSGUpstate/params.pickle",'rb') as f:
    param_set_list = pickle.load(f)
    
def mod_dist_gbar(model, mod_dict):
    for chan in mod_dict:
        if chan in model.Condset.D1.keys():
            #print("{} before: {}".format(chan, model.Condset.D1[chan]))
            model.Condset.D1[chan][model.param_cond.dist] *= mod_dict[chan]
            #print("modifying {} ".format(chan))
            #print("{} after: {}".format(chan, model.Condset.D1[chan]))


def setup_model(model, mod_dict, block_naf=False, filename=None,override_injection_current_list=None):
    model = importlib.import_module("moose_nerp.{}".format(model))
    # from IPython import embed; embed()
    from moose_nerp.prototypes import create_model_sim
    from moose_nerp.prototypes import spatiotemporalInputMapping as stim
    import moose

    if filename is not None:
        model.param_sim.fname = filename
    model.param_sim.save_txt = True
    model.param_sim.plot_vm = False
    model.param_sim.plot_current = True
    model.param_sim.plot_current_message = "getIk"
    model.spineYN = True
    model.calYN = True
    model.synYN = True
    model.SpineParams.explicitSpineDensity = 1e6
    if any("patch" in v for v in model.morph_file.values()):
        # model.SpineParams.spineParent = "570_3"
        model.clusteredparent = "570_3"
    if any("matrix" in v for v in model.morph_file.values()):
        # model.SpineParams.spineParent = "1157_3"
        model.clusteredparent = "1157_3"

    model.SpineParams.spineParent = model.clusteredparent  # "soma"
    modelname = model.__name__.split(".")[-1]
    model.param_syn._SynNMDA.Gbar = 10e-09 * mod_dict[modelname]["NMDA"]
    model.param_syn._SynNMDA.tau2 *= 2
    model.param_syn._SynNMDA.tau1 *= 2
    model.param_syn._SynAMPA.Gbar = 1e-9 * mod_dict[modelname]["AMPA"]
    mod_dist_gbar(model, mod_dict[modelname])
    if override_injection_current_list is not None:
        print(model.param_sim.injection_current)
        model.param_sim.injection_current = override_injection_current_list
        print(model.param_sim.injection_current)
    if block_naf:
        for k, v in model.Condset.D1.NaF.items():
            model.Condset.D1.NaF[k] = 0.0

    model.param_syn.SYNAPSE_TYPES.nmda.MgBlock.C = 1

    # create_model_sim.setupOptions(model)

    # create_model_sim.setupNeurons(model)

    # create_model_sim.setupOutput(model)

    return model

def iv_main(model, mod_dict, block_naf=False, filename=None,override_injection_current_list=None):
    print("filename: {}".format(filename))
    import numpy as np
    from moose_nerp.prototypes import create_model_sim
    from moose_nerp.prototypes import spatiotemporalInputMapping as stim
    import moose

    model = setup_model(model, mod_dict, block_naf=block_naf, filename=filename,override_injection_current_list=override_injection_current_list)
    model.param_sim.plot_current = False

    create_model_sim.setupOptions(model)
    create_model_sim.setupNeurons(model)
    create_model_sim.setupOutput(model)
    create_model_sim.setupStim(model)
    
    create_model_sim.runAll(model)

if __name__=='__main__':
    override_injection_current_list = list(np.linspace(-200e-12,200e-12,17))
    import sys
    args = sys.argv
    modeltype = args[1] if len(args)>1 else 'patch'
    print('modeltype = {}'.format(modeltype))
    if modeltype == 'patch':
        iv_main('D1PatchSample5',param_set_list[203],filename='patchsample5set203iv',override_injection_current_list=override_injection_current_list)
    if modeltype=='matrix':
        iv_main('D1MatrixSample2',param_set_list[45],filename='matrixsample2set45iv',override_injection_current_list=override_injection_current_list)
