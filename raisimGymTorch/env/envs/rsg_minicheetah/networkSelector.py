import numpy as np
import time
import torch


class NetworkSelector:
    def __init__(self, numEnv, device):
        self.value_run = torch.zeros(numEnv, 1)
        self.value_jump = torch.zeros(numEnv, 1)
        self.run_bool = torch.ones(numEnv, 1)
        self.step = torch.zeros(numEnv, 1)
        self.changeCounter = torch.zeros(numEnv, 1)
        self.obs_notNorm = torch.zeros(numEnv, 1)
        self.numEnv = numEnv
        self.device = device
        self.calculate_vector = -1 * np.ones(shape=(numEnv, 1), dtype=np.intc)
        self.old_bool = None


    def run_bool_function(self, value_run, value_jump, dones, obs_notNorm, selection_number):
        """ selects network"""
        not_done = torch.from_numpy(1 - dones).to(self.device)
        dones = torch.from_numpy(dones).to(self.device)
        # reset is done
        self.value_run = dones * torch.zeros(self.numEnv, 1, device=self.device) + not_done * self.value_run
        self.value_jump = dones * torch.zeros(self.numEnv, 1, device=self.device) + not_done * self.value_jump
        self.run_bool = dones * torch.ones(self.numEnv, 1, device=self.device) + not_done * self.run_bool
        self.step = dones * torch.zeros(self.numEnv, 1, device=self.device) + not_done * self.step
        self.changeCounter = dones * torch.zeros(self.numEnv, 1, device=self.device) + not_done * self.changeCounter


        self.step += torch.ones(self.numEnv, 1, device=self.device)

        if selection_number == 0:  # selection purely based on value
            self.value_run = value_run
            self.value_jump = value_jump
            self.run_bool = self.value_run > self.value_jump

        elif selection_number == 1:  # smoothing
            smoothing_factor = 0.999
            if self.step == 1:
                self.value_run = value_run
                self.value_jump = value_jump

            self.value_run = smoothing_factor * self.value_run + (1 - smoothing_factor) * value_run
            self.value_jump = smoothing_factor * self.value_jump + (1 - smoothing_factor) * value_jump
            self.run_bool = self.value_run > self.value_jump

        elif selection_number == 2:  # change after n steps better
            numChange = 5
            run_bool_step = value_run > value_jump
            changed = run_bool_step != self.run_bool
            self.changeCounter = changed * (self.changeCounter + torch.ones(self.numEnv, 1, device=self.device))  # add 1 if changed
            numChangeReached = self.changeCounter == numChange*torch.ones(self.numEnv, 1, device=self.device)
            numChangeNotReached = self.changeCounter != numChange*torch.ones(self.numEnv, 1, device=self.device)

            self.changeCounter = numChangeNotReached * self.changeCounter  # reset counter
            self.run_bool = numChangeReached * run_bool_step + numChangeNotReached * self.run_bool

        elif selection_number == 3:  # manually based on distance
            obs_ = obs_notNorm[:,-1:]  # only need distance
            run_bool = obs_.reshape(-1,1) > 0.65  # select jump network when observation shows robot close to hurdle  0.65
            number_Count = 30 + 1  #50 + 1

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

            cv_1 = (self.calculate_vector == -1) * (obs_ < 0.0) * -2  # compare above, sets to -2 if over hurdle
            cv_2 = (self.calculate_vector == -2) * (obs_ > 4)  # sets to 1 if out of observation range
            cv_3 = self.calculate_vector >= 1  # either cv_1 or cv_2 or cv_3 true, not multiple of them
            cv_4 = self.calculate_vector < number_Count

            calculate_vector_change = cv_1 + cv_2 + cv_3 * (cv_4 * (self.calculate_vector+1) + (1-cv_4) * -np.ones(self.calculate_vector.shape))
            self.calculate_vector = (calculate_vector_change != 0) * calculate_vector_change + (calculate_vector_change == 0) * self.calculate_vector

            self.run_bool = torch.from_numpy(run_bool * (self.calculate_vector < 0)).to(self.device)  # same as commented code, but faster


        elif selection_number == 4:  # only run
            self.run_bool = torch.ones(self.numEnv, 1)

        elif selection_number == 5:  # only jump
            self.run_bool = torch.zeros(self.numEnv, 1)

        if True:
            if self.old_bool is not None:
                old_bool = self.old_bool.cpu().numpy()
                run_bool = self.run_bool.cpu().numpy()
                if (old_bool.reshape(-1,1)[0,0] is np.bool_(True)) and (run_bool[0,0] is np.bool_(False)):
                    # if (old_bool.cpu().numpy().reshape(-1,1)[0,0] is np.bool_(True)) and (run_bool[0,0] is np.bool_(False)):
                    print("switch run -> jump")
                # elif (old_bool.cpu().numpy().reshape(-1,1)[0,0] is np.bool_(False)) and (run_bool[0,0] is np.bool_(True)):
                elif (old_bool.reshape(-1,1)[0,0] is np.bool_(False)) and (run_bool[0,0] is np.bool_(True)):
                    print("switch jump -> run")
            else:
                if self.run_bool[0,0] is np.bool_(False):
                    print("First selected network: jump")
                elif self.run_bool[0,0] is np.bool_(True):
                    print("First selected network: run")

        self.old_bool = self.run_bool

        return self.run_bool
