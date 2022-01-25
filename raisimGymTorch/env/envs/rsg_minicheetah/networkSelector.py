import numpy as np


def run_bool_function(obs_, act_dim_, output=False, old_bool=None):
    run_bool_next = obs_[:,-2:-1].reshape(-1,1) > 0.5
    run_bool_last = obs_[:,-1:].reshape(-1,1) < -0.3
    bool_env = np.logical_and(run_bool_next, run_bool_last)
    bool_actions = np.tile(bool_env, (1, act_dim_))

    if output:
        if (old_bool[0,0] is True) and (bool_actions[0,0] is False):
            print("switch run -> jump")
        elif (old_bool[0,0] is False) and (bool_actions[0,0] is True):
            print("switch jump -> run")
    return bool_actions
