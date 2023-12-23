#from __future__ import print_function, division

def main(p=1):
    import numpy as np
    state = np.random.get_state() 
    print(p,np.random.rand())
    return state

def multi_main():
    from multiprocessing.pool import Pool

    p = Pool(4, maxtasksperchild=1)
    # Apply main simulation varying cortical fractions:
    # cfs = ['FullTrialLowVariability', 'FullTrialHighVariability','FullTrialHigherVariability']
    cfs = [0,1,2,3]
    results = p.map(main, cfs)
    return dict(zip(cfs, results))


if __name__ == "__main__":
    print("runningmain")
    results = multi_main()