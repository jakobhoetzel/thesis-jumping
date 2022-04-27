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
        self.numEnv = numEnv
        self.device = device


    def run_bool_function(self, value_run, value_jump, dones, selection_number):
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

        elif selection_number == 3:  # only run
            self.run_bool = torch.ones(self.numEnv, 1)

        elif selection_number == 4:  # only jump
            self.run_bool = torch.zeros(self.numEnv, 1)

        return self.run_bool
