import numpy as np
import time
import torch


def run_bool_function(obs_, calculate_vector, output=False, old_bool=None, test_number=-1):
    """ selects jump network when close to hurdle, selects jump otherwise
    used when manager network is untrained"""
    if test_number == 0:
        print("No manual selection function available")
        return -1
    elif test_number == 2:
        rows, _ = obs_.shape
        return np.ones((rows, 1)).astype(bool), calculate_vector
    obs_ = obs_[:,-1:]  # only need distance
    run_bool = obs_.reshape(-1,1) > 0.65  # select jump network when observation shows robot close to hurdle
    # see observe in environment for definition (currently xPos_Hurdles_-ob.tail(1)(0)-0.15 )

    # if not run_bool_last or not run_bool_next:
    #     time.sleep(0.1)

    number_Count = 50 + 1
    # num_rows, _ = obs_.shape
    # cv_test = np.copy(calculate_vector)
    # run_bool_test = np.copy(run_bool)
    # for i in range(num_rows): #problem: reset
    #     if cv_test[i] == -1 and obs_[i]<0.0:  # jumped over hurdle
    #         cv_test[i] = -2
    #     elif cv_test[i]==-2 and obs_[i]>4: # after hurdle, but not in observation anymore
    #         cv_test[i] = 1
    #     elif cv_test[i] >= 1:  # after hurdle, but not in observation anymore
    #         if cv_test[i]<number_Count:
    #             cv_test[i] += 1
    #         else:
    #             cv_test[i] = -1 # further away
    #
    #     if cv_test[i] >= 0:
    #         run_bool_test[i] = False

    cv_1 = (calculate_vector == -1) * (obs_ < 0.0) * -2
    cv_2 = (calculate_vector == -2) * (obs_ > 4)
    cv_3 = calculate_vector >= 1  # either cv_1 or cv_2 or cv_3 true, not multiple
    cv_4 = calculate_vector < number_Count

    calculate_vector_change = cv_1 + cv_2 + cv_3 * (cv_4 * (calculate_vector+1) + (1-cv_4) * -np.ones(calculate_vector.shape))
    calculate_vector = (calculate_vector_change != 0) * calculate_vector_change + (calculate_vector_change == 0) * calculate_vector

    run_bool = run_bool * (calculate_vector < 0)  # same as commented code, but faster
    # arrayEqual = np.array_equal(run_bool, run_bool_test) and np.array_equal(calculate_vector, cv_test)
    # if not arrayEqual:
    #     print("Error in new network selector calculation")

    if output:
        if old_bool is not None:
            old_bool = old_bool.numpy()
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
    return run_bool, calculate_vector

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
