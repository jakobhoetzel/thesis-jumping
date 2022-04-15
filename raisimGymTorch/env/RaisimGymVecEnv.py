# //----------------------------//
# // This file is part of RaiSim//
# // Copyright 2020, RaiSim Tech//
# //----------------------------//

import numpy as np
import platform
import os


class RaisimGymVecEnv:

    def __init__(self, impl, cfg, normalize_ob=True, seed=0, normalize_rew=True, clip_obs=10., sensor_dim=0):
        if platform.system() == "Darwin":
            os.environ['KMP_DUPLICATE_LIB_OK']='True'
        self.normalize_ob = normalize_ob
        self.normalize_rew = normalize_rew
        self.clip_obs = clip_obs
        self.wrapper = impl
        self.wrapper.init()
        self.num_obs = self.wrapper.getObDim()
        self.num_sens = sensor_dim
        self.num_robotState = self.wrapper.getRobotStateDim()
        self.num_acts = self.wrapper.getActionDim()
        self._observation = np.zeros([self.num_envs, self.num_obs], dtype=np.float32)
        self._robotState = np.zeros([self.num_envs, self.num_robotState], dtype=np.float32)
        self.obs_rms_run = RunningMeanStd(shape=[self.num_envs, self.num_obs])
        self.obs_rms_jump = RunningMeanStd(shape=[self.num_envs, self.num_obs])
        self.obs_rms_manager = RunningMeanStd(shape=[self.num_envs, self.num_obs])
        self._reward = np.zeros(self.num_envs, dtype=np.float32)
        self._done = np.zeros(self.num_envs, dtype=np.bool)
        self.rewards = [[] for _ in range(self.num_envs)]

    def seed(self, seed=None):
        self.wrapper.setSeed(seed)

    def set_command(self, command, testNumber=0):
        self.wrapper.setCommand(command, testNumber)

    def turn_on_visualization(self):
        self.wrapper.turnOnVisualization()

    def turn_off_visualization(self):
        self.wrapper.turnOffVisualization()

    def start_video_recording(self, file_name):
        self.wrapper.startRecordingVideo(file_name)

    def stop_video_recording(self):
        self.wrapper.stopRecordingVideo()

    def step(self, action, run_bool, manager_training):
        self.wrapper.step(action, self._reward, self._done, run_bool[:,0].astype(bool), manager_training)
        return self._reward.copy(), self._done.copy()

    def load_scaling(self, dir_name_run, iteration_run, dir_name_jump, iteration_jump, dir_name_manager, iteration_manager, count=1e5, one_directory=True):
        if not one_directory:
            mean_file_name_run = dir_name_run + "/mean" + str(iteration_run) + ".csv"
            var_file_name_run = dir_name_run + "/var" + str(iteration_run) + ".csv"
            mean_file_name_jump = dir_name_jump + "/mean" + str(iteration_jump) + ".csv"
            var_file_name_jump = dir_name_jump + "/var" + str(iteration_jump) + ".csv"
            mean_file_name_manager = dir_name_manager + "/mean" + str(iteration_manager) + ".csv"
            var_file_name_manager = dir_name_manager + "/var" + str(iteration_manager) + ".csv"
        else:
            mean_file_name_run = dir_name_run + "/meanRun" + str(iteration_run) + ".csv"
            var_file_name_run = dir_name_run + "/varRun" + str(iteration_run) + ".csv"
            mean_file_name_jump = dir_name_jump + "/meanJump" + str(iteration_jump) + ".csv"
            var_file_name_jump = dir_name_jump + "/varJump" + str(iteration_jump) + ".csv"
            mean_file_name_manager = dir_name_manager + "/meanManager" + str(iteration_manager) + ".csv"
            var_file_name_manager = dir_name_manager + "/varManager" + str(iteration_manager) + ".csv"
        self.obs_rms_run.count = count
        self.obs_rms_jump.count = count
        self.obs_rms_manager.count = count
        for i in range(self.num_envs):
            self.obs_rms_run.mean[i] = np.loadtxt(mean_file_name_run, dtype=np.float32)
            self.obs_rms_run.var[i] = np.loadtxt(var_file_name_run, dtype=np.float32)
            self.obs_rms_jump.mean[i] = np.loadtxt(mean_file_name_jump, dtype=np.float32)
            self.obs_rms_jump.var[i] = np.loadtxt(var_file_name_jump, dtype=np.float32)
            # self.obs_rms_manager.mean[i] = np.loadtxt(mean_file_name_manager, dtype=np.float32)
            # self.obs_rms_manager.var[i] = np.loadtxt(var_file_name_manager, dtype=np.float32)

    def save_scaling(self, dir_name, iteration):
        mean_file_name_run = dir_name + "/meanRun" + iteration + ".csv"
        var_file_name_run = dir_name + "/varRun" + iteration + ".csv"
        mean_file_name_jump = dir_name + "/meanJump" + iteration + ".csv"
        var_file_name_jump = dir_name + "/varJump" + iteration + ".csv"
        # mean_file_name_manager = dir_name + "/meanManager" + iteration + ".csv"
        # var_file_name_manager = dir_name + "/varManager" + iteration + ".csv"
        # mean_file_name_manager = dir_name + "/mean" + iteration + ".csv"
        # var_file_name_manager = dir_name + "/var" + iteration + ".csv"
        np.savetxt(mean_file_name_run, self.obs_rms_run.mean[0])
        np.savetxt(var_file_name_run, self.obs_rms_run.var[0])
        np.savetxt(mean_file_name_jump, self.obs_rms_jump.mean[0])
        np.savetxt(var_file_name_jump, self.obs_rms_jump.var[0])
        # np.savetxt(mean_file_name_manager, self.obs_rms_manager.mean[0])
        # np.savetxt(var_file_name_manager, self.obs_rms_manager.var[0])

    def observe(self, update_mean=True, update_manager=False):
        self.wrapper.observe(self._observation)

        if self.normalize_ob:
            if update_mean:
                if update_manager:
                    self.obs_rms_manager.update(self._observation)
                else:
                    self.obs_rms_run.update(self._observation)  # slight 'cheat' as only one is in use
                    self.obs_rms_jump.update(self._observation)

            return self._normalize_observation(self._observation), self._observation.copy()
        else:
            return self._observation.copy()

    def getRobotState(self):
        self.wrapper.getRobotState(self._robotState)
        return self._robotState.copy()

    def reset(self):
        self._reward = np.zeros(self.num_envs, dtype=np.float32)
        self.wrapper.reset()

    def _normalize_observation(self, obs):
        if self.normalize_ob:

            obs_run = np.clip((obs - self.obs_rms_run.mean) / np.sqrt(self.obs_rms_run.var + 1e-8), -self.clip_obs,
                              self.clip_obs)
            obs_jump = np.clip((obs - self.obs_rms_jump.mean) / np.sqrt(self.obs_rms_jump.var + 1e-8), -self.clip_obs,
                              self.clip_obs)
            obs_manager = np.clip((obs - self.obs_rms_manager.mean) / np.sqrt(self.obs_rms_manager.var + 1e-8), -self.clip_obs,
                              self.clip_obs)
            return obs_run, obs_jump, obs_manager
        else:
            return obs, obs, obs

    def reset_and_update_info(self):
        return self.reset(), self._update_epi_info()

    def _update_epi_info(self):
        info = [{} for _ in range(self.num_envs)]

        for i in range(self.num_envs):
            eprew = sum(self.rewards[i])
            eplen = len(self.rewards[i])
            epinfo = {"r": eprew, "l": eplen}
            info[i]['episode'] = epinfo
            self.rewards[i].clear()

        return info

    def close(self):
        self.wrapper.close()

    def curriculum_callback(self, iter):
        self.wrapper.curriculumUpdate(iter)

    def go_straight_controller(self):
        self.wrapper.go_straight_controller()

    def get_step_data_tag(self):
        return self.wrapper.getStepDataTag()

    def get_step_data(self, data_size, data_mean, data_var, data_min, data_max):
        return self.wrapper.getStepData(data_size, data_mean, data_var, data_min, data_max)

    def printTest(self):
        self.wrapper.printTest()

    @property
    def num_envs(self):
        return self.wrapper.getNumOfEnvs()

    @property
    def extra_info_names(self):
        return self._extraInfoNames


class RunningMeanStd(object):
    def __init__(self, epsilon=1e-4, shape=()):
        """
        calulates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

        :param epsilon: (float) helps with arithmetic issues
        :param shape: (tuple) the shape of the data stream's output
        """
        self.mean = np.zeros(shape, 'float32')
        self.var = np.ones(shape, 'float32')
        self.count = epsilon

    def update(self, arr):
        batch_mean = np.mean(arr, axis=0)
        batch_var = np.var(arr, axis=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * (self.count * batch_count / (self.count + batch_count))
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

