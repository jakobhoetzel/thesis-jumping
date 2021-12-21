from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import rsg_minicheetah
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver, load_param, tensorboard_launcher
import os
import math
import time
import raisimGymTorch.algo.ppo.module as ppo_module
import raisimGymTorch.algo.ppo.ppo as PPO
import torch.nn as nn
import numpy as np
import torch
import datetime
import argparse


# task specification
task_name = "minicheetah_locomotion"  # "~~~/raisimGymTorch/data/"+task_name: log directory

# configuration
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', help='set mode either train or retrain', type=str, default='train')
parser.add_argument('-w', '--weight', help='pre-trained weight path', type=str, default='')
args = parser.parse_args()
mode = args.mode  # 'train' or 'retrain'
weight_path = args.weight

# check if gpu is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# directories
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../../../.."

# config
cfg = YAML().load(open(task_path + "/cfg.yaml", 'r'))

# create environment from the configuration file
env = VecEnv(rsg_minicheetah.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)), cfg['environment'])

# shortcuts
ob_dim = env.num_obs
robotState_dim = env.num_robotState
act_dim = env.num_acts

# Training
n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])  # 400
total_steps = n_steps * env.num_envs  # 40000

avg_rewards = []

actor = ppo_module.Actor(ppo_module.MLP(cfg['architecture']['policy_net'], nn.LeakyReLU, ob_dim + robotState_dim, act_dim),
                         ppo_module.MultivariateGaussianDiagonalCovariance(act_dim, 1.0),  # 1.0
                         device)
critic = ppo_module.Critic(ppo_module.MLP(cfg['architecture']['value_net'], nn.LeakyReLU, ob_dim + robotState_dim, 1),
                           device)
stateEstimator = ppo_module.StateEstimator(ppo_module.MLP(cfg['architecture']['estimator_net'], nn.LeakyReLU, ob_dim, robotState_dim),
                                           device)

saver = ConfigurationSaver(log_dir=home_path + "/raisimGymTorch/data/"+task_name,  # save environment and configuration data.
                           save_items=[task_path + "/cfg.yaml", task_path + "/Environment.hpp", task_path + "/MinicheetahController.hpp",
                                       task_path + "/runner.py", task_path + "/../../VectorizedEnvironment.hpp",
                                       task_path + "/../../RaisimGymVecEnv.py", task_path + "/../../raisim_gym.cpp",
                                       task_path + "/../../../algo/ppo/module.py", task_path + "/../../../../rsc/mini_cheetah/mini-cheetah-vision-v1.5.urdf",
                                       task_path + "/../../../algo/ppo/ppo.py", task_path + "/../../../algo/ppo/storage.py"])
tensorboard_launcher(saver.data_dir+"/..")  # press refresh (F5) after the first ppo update

ppo = PPO.PPO(actor=actor,
              critic=critic,
              estimator=stateEstimator,
              num_envs=cfg['environment']['num_envs'],
              num_transitions_per_env=n_steps,
              num_learning_epochs=4,
              gamma=0.99,
              lam=0.95,
              learning_rate=5e-4,
              entropy_coef=0.01,
              num_mini_batches=8,
              device=device,
              log_dir=saver.data_dir,
              shuffle_batch=False,
              )

data_tags = env.get_step_data_tag()

scheduler = torch.optim.lr_scheduler.MultiStepLR(ppo.optimizer, milestones=[2000], gamma=0.333333)

if mode == 'retrain':
    load_param(weight_path, env, actor, critic, stateEstimator, ppo.optimizer, saver.data_dir)

max_iteration = 5000 + 1

for update in range(max_iteration):
    start = time.time()
    env.reset()
    reward_ll_sum = 0
    done_sum = 0
    average_dones = 0.

    env.curriculum_callback(update)

    if update % cfg['environment']['eval_every_n'] == 0:
        print("Visualizing and evaluating the current policy")
        torch.save({
            'actor_architecture_state_dict': actor.architecture.state_dict(),
            'actor_distribution_state_dict': actor.distribution.state_dict(),
            'critic_architecture_state_dict': critic.architecture.state_dict(),
            'estimator_architecture_state_dict': stateEstimator.architecture.state_dict(),
            'optimizer_state_dict': ppo.optimizer.state_dict(),
        }, saver.data_dir+"/full_"+str(update)+'.pt')
        actor.save_deterministic_graph(saver.data_dir + "/actor_" + str(update) + ".pt", torch.rand(1, ob_dim + robotState_dim).cpu())
        stateEstimator.save_deterministic_graph(saver.data_dir + "/estimator_" + str(update) + ".pt", torch.rand(1, ob_dim).cpu())

        # we create another graph just to demonstrate the save/load method
        loaded_graph = ppo_module.MLP(cfg['architecture']['policy_net'], torch.nn.LeakyReLU, ob_dim + robotState_dim, act_dim)
        loaded_graph.load_state_dict(torch.load(saver.data_dir+"/full_"+str(update)+'.pt')['actor_architecture_state_dict'])

        env.turn_on_visualization()
        env.start_video_recording(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "policy_"+str(update)+'.mp4')
        time.sleep(1)

        for step in range(n_steps*1):  # n_steps*2
            frame_start = time.time()
            obs = env.observe(False)  # don't compute rms
            robotState = env.getRobotState()
            est_out = stateEstimator.predict(torch.from_numpy(obs).to(device))
            concatenated_obs_actor = np.concatenate((obs, est_out.cpu().detach().numpy()), axis=1)
            concatenated_obs_critic = np.concatenate((obs, robotState), axis=1)
            action = ppo.observe(concatenated_obs_actor)
            action_ll, _ = actor.sample(torch.from_numpy(concatenated_obs_actor).to(device))  # stochastic action
            # action_ll = loaded_graph.architecture(torch.from_numpy(obs).cpu())

            reward_ll, dones = env.step(action_ll.cpu().numpy())  # in stochastic action case
            frame_end = time.time()
            wait_time = cfg['environment']['control_dt'] - (frame_end-frame_start)
            if wait_time > 0.:
                time.sleep(wait_time)

        env.stop_video_recording()
        env.turn_off_visualization()

        env.reset()
        env.save_scaling(saver.data_dir, str(update))

    data_size = 0
    data_mean = np.zeros(shape=(len(data_tags), 1), dtype=np.double)
    data_square_sum = np.zeros(shape=(len(data_tags), 1), dtype=np.double)
    data_min = np.inf * np.ones(shape=(len(data_tags), 1), dtype=np.double)
    data_max = -np.inf * np.ones(shape=(len(data_tags), 1), dtype=np.double)

    # actual training
    for step in range(n_steps):
        obs = env.observe()
        robotState = env.getRobotState()
        est_out = stateEstimator.predict(torch.from_numpy(obs).to(device))
        concatenated_obs_actor = np.concatenate((obs, est_out.cpu().detach().numpy()), axis=1)
        concatenated_obs_critic = np.concatenate((obs, robotState), axis=1)
        action = ppo.observe(concatenated_obs_actor)
        reward, dones = env.step(action)
        ppo.step(value_obs=concatenated_obs_critic, est_in=obs, robotState=robotState, rews=reward, dones=dones)
        done_sum = done_sum + sum(dones)
        reward_ll_sum = reward_ll_sum + sum(reward)
        data_size = env.get_step_data(data_size, data_mean, data_square_sum, data_min, data_max)

    data_std = np.sqrt((data_square_sum - data_size * data_mean * data_mean) / (data_size - 1 + 1e-16))

    if update % 20 == 0:
        for data_id in range(len(data_tags)):
            ppo.writer.add_scalar(data_tags[data_id]+'/mean', data_mean[data_id], global_step=update)
            ppo.writer.add_scalar(data_tags[data_id]+'/std', data_std[data_id], global_step=update)
            ppo.writer.add_scalar(data_tags[data_id]+'/min', data_min[data_id], global_step=update)
            ppo.writer.add_scalar(data_tags[data_id]+'/max', data_max[data_id], global_step=update)

    # take st step to get value obs
    obs = env.observe()
    robotState = env.getRobotState()
    est_out = stateEstimator.predict(torch.from_numpy(obs).to(device))
    concatenated_obs_actor = np.concatenate((obs, est_out.cpu().detach().numpy()), axis=1)
    concatenated_obs_critic = np.concatenate((obs, robotState), axis=1)
    ppo.update(actor_obs=concatenated_obs_actor, value_obs=concatenated_obs_critic, log_this_iteration=update % 20 == 0, update=update)
    average_ll_performance = reward_ll_sum / total_steps  # average reward per step per environment
    average_dones = done_sum / total_steps
    avg_rewards.append(average_ll_performance)


    actor.distribution.enforce_minimum_std((torch.ones(12)*0.25).to(device))

    # env.curriculum_callback(update) # at the beginning now

    end = time.time()

    scheduler.step()

    print('----------------------------------------------------')
    print('{:>6} /{:>6}th iteration'.format(update+1, max_iteration))  # {:>6} means printing at least 6 character space and the string is printed from the right. {:<6} starting from the left
    print('{:<40} {:>6}'.format("average ll reward: ", '{:0.10f}'.format(average_ll_performance)))
    print('{:<40} {:>6}'.format("dones: ", '{:0.6f}'.format(average_dones)))
    print('{:<40} {:>6}'.format("time elapsed in this iteration: ", '{:6.4f}'.format(end - start)))
    print('{:<40} {:>6}'.format("fps: ", '{:6.0f}'.format(total_steps / (end - start))))
    print('{:<40} {:>6}'.format("real time factor: ", '{:6.0f}'.format(total_steps / (end - start)
                                                                       * cfg['environment']['control_dt'])))
    print('std: ')
    print(np.exp(actor.distribution.std.cpu().detach().numpy()))
    print('----------------------------------------------------\n')