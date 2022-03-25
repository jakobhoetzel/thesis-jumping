import numpy as np
import time
import torch


def run_bool_function(obs_, output=False, old_bool=None):
    """ selects jump network when close to hurdle, selects jump otherwise
    used when manager network is untrained"""
    run_bool = obs_[:,-1:].reshape(-1,1) > 0.65  # select jump network when observation shows robot close to hurdle
    # see observe in environment for definition

    # if not run_bool_last or not run_bool_next:
    #     time.sleep(0.1)

    if output:
        if old_bool is not None:
            if (old_bool.reshape(-1,1)[0,0] is np.bool_(True)) and (run_bool[0,0] is np.bool_(False)):
            # if (old_bool.cpu().numpy().reshape(-1,1)[0,0] is np.bool_(True)) and (run_bool[0,0] is np.bool_(False)):
                print("switch run -> jump")
            # elif (old_bool.cpu().numpy().reshape(-1,1)[0,0] is np.bool_(False)) and (run_bool[0,0] is np.bool_(True)):
            elif (old_bool.reshape(-1,1)[0,0] is np.bool_(False)) and (run_bool[0,0] is np.bool_(True)):
                print("switch jump -> run")
        else:
            if run_bool[0,0] is np.bool_(False):
                print("First selected network: jump")
            elif run_bool[0,0] is np.bool_(True):
                print("First selected network: run")
    return run_bool

def run_bool_function_0(obs_):
    """ always runner network"""
    rows, _ = obs_.shape
    run_bool = np.zeros((rows, 1)).astype(bool)

    return run_bool

def run_bool_function_1(obs_):
    """ always jump network"""
    rows, _ = obs_.shape
    run_bool = np.ones((rows, 1)).astype(bool)

    return run_bool
