import numpy as np
import time
import torch


def run_bool_function(obs_, output=False, old_bool=None):
    run_bool_next = obs_[:,-2:-1].reshape(-1,1) > 0.7
    run_bool_last = obs_[:,-1:].reshape(-1,1) < -0.3
    run_bool = np.logical_and(run_bool_next, run_bool_last)

    # if not run_bool_last or not run_bool_next:
    #     time.sleep(0.1)

    if output:
        if old_bool is not None:
            if (old_bool.cpu().numpy().reshape(-1,1)[0,0] is np.bool_(True)) and (run_bool[0,0] is np.bool_(False)):
                print("switch run -> jump")
            elif (old_bool.cpu().numpy().reshape(-1,1)[0,0] is np.bool_(False)) and (run_bool[0,0] is np.bool_(True)):
                print("switch jump -> run")
        else:
            if run_bool[0,0] is np.bool_(False):
                print("First selected network: jump")
            elif run_bool[0,0] is np.bool_(True):
                print("First selected network: run")
    return run_bool
