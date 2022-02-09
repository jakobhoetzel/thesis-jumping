from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import rsg_minicheetah
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver, load_param, tensorboard_launcher
from networkSelector import run_bool_function
from freeze import freeze_actors, freeze_manager
import os
import math
import time
import raisimGymTorch.algo.ppo.module as ppo_module
import raisimGymTorch.algo.ppo.ppo as PPO
import raisimGymTorch.algo.ppo.IdentityLearning as IL
import torch.nn as nn
import numpy as np
import torch
import datetime
import argparse

# from torch.distributions import Categorical
# m = Categorical(torch.tensor([ 0.25, 0.25, 0.25, 0.25 ]))
# ms = m.sample()  # equal probability of 0, 1, 2, 3
# msi = ms.item()
# exit()

# task specification
task_name = "minicheetah_locomotion"  # "~~~/raisimGymTorch/data/"+task_name: log directory

# configuration
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', help='set mode either train or retrain', type=str, default='train')
parser.add_argument('-w', '--weight', help='pre-trained weight path', type=str, default='')
parser.add_argument('-n', '--runNumber', help='number of the run', type=str, default='')
args = parser.parse_args()
mode = args.mode  # 'train' or 'retrain'
runNumber = args.runNumber
#weight_path = args.weight
weight_path_run = "../../../data/minicheetah_locomotion/baselineRun2/full_5000.pt"
iteration_number_run = weight_path_run.rsplit('/', 1)[1].split('_', 1)[1].rsplit('.', 1)[0]
weight_dir_run = weight_path_run.rsplit('/', 1)[0] + '/'

weight_path_jump = "../../../data/minicheetah_locomotion/baselineJump0/full_5000.pt"
iteration_number_jump = weight_path_jump.rsplit('/', 1)[1].split('_', 1)[1].rsplit('.', 1)[0]
weight_dir_jump = weight_path_jump.rsplit('/', 1)[0] + '/'

# check if gpu is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# directories
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../../../.."

# config
cfg = YAML().load(open(task_path + "/cfg.yaml", 'r'))

# if runNumber == 0:
#     cfg['environment']['ter_curriculum_factor'] = 0.0
# elif runNumber == 1:
#     cfg['environment']['ter_curriculum_factor'] = 0.2
# elif runNumber == 2:
#     cfg['environment']['ter_curriculum_factor'] = 0.4

# create environment from the configuration file
env = VecEnv(rsg_minicheetah.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)), cfg['environment'])

# shortcuts
ob_dim = env.num_obs  # including sensor
robotState_dim = env.num_robotState
act_dim = env.num_acts
sensor_dim = 3

# Training
n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])  # 400
total_steps = n_steps * env.num_envs  # 40000

avg_rewards = []

savedWeights_run = torch.load(weight_path_run)['actor_architecture_state_dict']['architecture.0.weight']
savedWeights_run_obs = savedWeights_run[:, :ob_dim-sensor_dim]
savedWeights_run_robotState = savedWeights_run[:, -robotState_dim:]
addedWeights_run = torch.randn(512,sensor_dim,device=device)*0.000001
combinedWeights_run = torch.cat([savedWeights_run_obs, addedWeights_run, savedWeights_run_robotState], dim=1)
savedDict_run = torch.load(weight_path_run)['actor_architecture_state_dict']
savedDict_run['architecture.0.weight'] = torch.nn.Parameter(combinedWeights_run)
mlpActor_run = ppo_module.MLP(cfg['architecture']['policy_net'], torch.nn.LeakyReLU, ob_dim + robotState_dim, act_dim).to(device)
mlpActor_run.load_state_dict(savedDict_run)
actor_run = ppo_module.Actor(mlpActor_run,
                             ppo_module.MultivariateGaussianDiagonalCovariance(act_dim, 1.0),  # 1.0
                             device)

savedWeights_jump = torch.load(weight_path_jump)['actor_architecture_state_dict']['architecture.0.weight']
savedDict_jump = torch.load(weight_path_jump)['actor_architecture_state_dict']
savedDict_jump['architecture.0.weight'] = torch.nn.Parameter(savedWeights_jump)
mlpActor_jump = ppo_module.MLP(cfg['architecture']['policy_net'], torch.nn.LeakyReLU, ob_dim + robotState_dim, act_dim).to(device)
mlpActor_jump.load_state_dict(savedDict_jump)
actor_jump = ppo_module.Actor(mlpActor_jump,
                              ppo_module.MultivariateGaussianDiagonalCovariance(act_dim, 1.0),  # 1.0
                              device)

actor_manager = ppo_module.Manager(ppo_module.MLP(cfg['architecture']['manager_net'], nn.LeakyReLU, ob_dim + robotState_dim, 2, softmax=True),
                         device)
critic_run = ppo_module.Critic(ppo_module.MLP(cfg['architecture']['value_net'], nn.LeakyReLU, ob_dim + robotState_dim, 1),
                              device)
critic_jump = ppo_module.Critic(ppo_module.MLP(cfg['architecture']['value_net'], nn.LeakyReLU, ob_dim + robotState_dim, 1),
                                device)
critic_manager = ppo_module.Critic(ppo_module.MLP(cfg['architecture']['value_net'], nn.LeakyReLU, ob_dim + robotState_dim, 1),
                                device)
mlpEstimator = ppo_module.MLP(cfg['architecture']['estimator_net'], torch.nn.LeakyReLU, ob_dim - sensor_dim, robotState_dim).to(device)
mlpEstimator.load_state_dict(torch.load(weight_path_run)['estimator_architecture_state_dict']) # or weight_path_jump??
stateEstimator = ppo_module.StateEstimator(mlpEstimator,
                                           device)

# actor_test = ppo_module.MLP(cfg['architecture']['policy_net'], torch.nn.LeakyReLU, ob_dim - sensor_dim + robotState_dim, act_dim).to(device)
# actor_test.load_state_dict(torch.load(weight_path)['actor_architecture_state_dict'])

saver = ConfigurationSaver(log_dir=home_path + "/raisimGymTorch/data/"+task_name,  # save environment and configuration data.
                           save_items=[task_path + "/cfg.yaml", task_path + "/Environment.hpp", task_path + "/MinicheetahController.hpp",
                                       task_path + "/runner.py", task_path + "/../../VectorizedEnvironment.hpp",
                                       task_path + "/../../RaisimGymVecEnv.py", task_path + "/../../raisim_gym.cpp",
                                       task_path + "/../../../algo/ppo/module.py", task_path + "/../../../../rsc/mini_cheetah/mini-cheetah-vision-v1.5.urdf",
                                       task_path + "/../../../algo/ppo/ppo.py", task_path + "/../../../algo/ppo/storage.py"])
file = open(saver.data_dir + "/README.txt", "w") # only for logging
file.write('Number of the run = ' + str(runNumber) + '\n')
file.close()
tensorboard_launcher(saver.data_dir+"/..")  # press refresh (F5) after the first ppo update

ppo = PPO.PPO(actor_run=actor_run,
              actor_jump=actor_jump,
              actor_manager=actor_manager,
              critic_run=critic_run,
              critic_jump=critic_jump,
              critic_manager=critic_manager,
              estimator=stateEstimator,
              num_envs=cfg['environment']['num_envs'],
              num_transitions_per_env=n_steps,
              num_learning_epochs=4,
              clip_param=0.2,  # 0.2
              gamma=0.99,
              lam=0.95,
              learning_rate=5e-4,  # 5e-4
              entropy_coef=0.01,
              num_mini_batches=8,
              device=device,
              log_dir=saver.data_dir,
              shuffle_batch=False,
              manager_training=False,
              )

data_tags = env.get_step_data_tag()

scheduler = torch.optim.lr_scheduler.MultiStepLR(ppo.optimizer, milestones=[2000], gamma=0.333333)

# if mode == 'retrain':
#     load_param(weight_path, env, actor_run, actor_jump, critic_run, critic_jump, stateEstimator, ppo.optimizer, saver.data_dir)

max_iteration = 10000 + 1 #5000+1

env.load_scaling(weight_dir_run, int(iteration_number_run), 1e8) # 1e8 -> less disruption when retraining #TODO: different scaling for different networks

ppo.set_manager_training(True)
# ppo.set_manager_training(False)
# freeze_manager(ppo)
# freeze_actors(ppo)

for update in range(max_iteration):
    start = time.time()
    env.reset()
    reward_ll_sum = 0
    done_sum = 0
    average_dones = 0.

    env.curriculum_callback(update//2 + 5000)  # start with half height

    if update % cfg['environment']['eval_every_n'] == 0:
        print("Visualizing and evaluating the current policy")
        torch.save({
            'actor_run_architecture_state_dict': actor_run.architecture.state_dict(),
            'actor_run_distribution_state_dict': actor_run.distribution.state_dict(),
            'actor_jump_architecture_state_dict': actor_jump.architecture.state_dict(),
            'actor_jump_distribution_state_dict': actor_jump.distribution.state_dict(),
            'actor_manager_architecture_state_dict': actor_manager.architecture.state_dict(),
            'critic_run_architecture_state_dict': critic_run.architecture.state_dict(),
            'critic_jump_architecture_state_dict': critic_jump.architecture.state_dict(),
            'estimator_architecture_state_dict': stateEstimator.architecture.state_dict(),
            'optimizer_state_dict': ppo.optimizer.state_dict(),
        }, saver.data_dir+"/full_"+str(update)+'.pt')
        actor_run.save_deterministic_graph(saver.data_dir + "/actor_run_" + str(update) + ".pt", torch.rand(1, ob_dim + robotState_dim).cpu())
        actor_jump.save_deterministic_graph(saver.data_dir + "/actor_jump_" + str(update) + ".pt", torch.rand(1, ob_dim + robotState_dim).cpu())
        actor_jump.save_deterministic_graph(saver.data_dir + "/actor_manager_" + str(update) + ".pt", torch.rand(1, ob_dim + robotState_dim).cpu())
        stateEstimator.save_deterministic_graph(saver.data_dir + "/estimator_" + str(update) + ".pt", torch.rand(1, ob_dim-sensor_dim).cpu())

        temp_obs = np.ones((cfg['environment']['num_envs'], ob_dim + robotState_dim), dtype=np.float32)  # to see if input network changes
        temp_action_run = actor_run.architecture.architecture(torch.from_numpy(temp_obs).to(device)).detach()
        temp_action_jump = actor_jump.architecture.architecture(torch.from_numpy(temp_obs).to(device)).detach()
        temp_action_manager = actor_manager.architecture.architecture(torch.from_numpy(temp_obs).to(device)).detach()
        np.savetxt("ones_action_run_beg.csv", temp_action_run.cpu().numpy(), delimiter=",")
        np.savetxt("ones_action_jump_beg.csv", temp_action_jump.cpu().numpy(), delimiter=",")
        np.savetxt("ones_action_manager_beg.csv", temp_action_manager.cpu().numpy(), delimiter=",")

        # we create another graph just to demonstrate the save/load method
        # loaded_graph = ppo_module.MLP(cfg['architecture']['policy_net'], torch.nn.LeakyReLU, ob_dim + robotState_dim, act_dim)
        # loaded_graph.load_state_dict(torch.load(saver.data_dir+"/full_"+str(update)+'.pt')['actor_architecture_state_dict'])

        env.turn_on_visualization()
        env.start_video_recording(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "policy_"+str(update)+'.mp4')
        time.sleep(1)

        for step in range(n_steps*1):  # n_steps*2
            frame_start = time.time()
            obs = env.observe(False)  # don't compute rms
            obs_estimator = obs[:,:ob_dim-sensor_dim]
            robotState = env.getRobotState()
            est_out = stateEstimator.predict(torch.from_numpy(obs_estimator).to(device))
            concatenated_obs_actor = np.concatenate((obs, est_out.cpu().detach().numpy()), axis=1)
            concatenated_obs_critic = np.concatenate((obs, robotState), axis=1)
            action = ppo.observe(concatenated_obs_actor)

            # action_ll, _ = actor.sample(torch.from_numpy(concatenated_obs_actor).to(device))  # stochastic action
            # action_ll = loaded_graph.architecture(torch.from_numpy(obs).cpu())

            reward_ll, dones = env.step(action)  # in stochastic action case
            env.go_straight_controller()
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
        obs_estimator = obs[:,:ob_dim-sensor_dim]
        robotState = env.getRobotState()
        est_out = stateEstimator.predict(torch.from_numpy(obs_estimator).to(device))
        concatenated_obs_actor = np.concatenate((obs, est_out.cpu().detach().numpy()), axis=1)
        concatenated_obs_critic = np.concatenate((obs, robotState), axis=1)
        action = ppo.observe(concatenated_obs_actor)
        reward, dones = env.step(action)
        env.go_straight_controller()
        ppo.step(value_obs=concatenated_obs_critic, est_in=obs_estimator, robotState=robotState, rews=reward, dones=dones)
        done_sum = done_sum + np.sum(dones)
        reward_ll_sum = reward_ll_sum + np.sum(reward)
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
    obs_estimator = obs[:,:ob_dim-sensor_dim]
    robotState = env.getRobotState()
    est_out = stateEstimator.predict(torch.from_numpy(obs_estimator).to(device))
    concatenated_obs_actor = np.concatenate((obs, est_out.cpu().detach().numpy()), axis=1)
    concatenated_obs_critic = np.concatenate((obs, robotState), axis=1)

    # a0 = list(actor_run.parameters())[0].grad
    # print("a0: ", a0)
    # a1 = list(actor_jump.parameters())[0].grad
    # print("a1: ", a1)
    # a2 = list(actor_manager.parameters())[0].grad
    # print("a2: ", a2)

    ppo.update(actor_obs=concatenated_obs_actor, value_obs=concatenated_obs_critic, log_this_iteration=update % 20 == 0, update=update)
    average_ll_performance = reward_ll_sum / total_steps  # average reward per step per environment
    average_dones = done_sum / total_steps
    avg_rewards.append(average_ll_performance)


    actor_run.distribution.enforce_minimum_std(torch.ones(12, device=device)*0.25)
    actor_jump.distribution.enforce_minimum_std(torch.ones(12, device=device)*0.25)

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
    print(np.exp(actor_run.distribution.std.cpu().detach().numpy()))
    print(np.exp(actor_jump.distribution.std.cpu().detach().numpy()))
    print('----------------------------------------------------\n')

    if update % cfg['environment']['eval_every_n'] == 0:
        temp_obs = np.ones((cfg['environment']['num_envs'], ob_dim + robotState_dim), dtype=np.float32)  # to see if input network changes
        temp_action_run = actor_run.architecture.architecture(torch.from_numpy(temp_obs).to(device)).detach()
        temp_action_jump = actor_jump.architecture.architecture(torch.from_numpy(temp_obs).to(device)).detach()
        temp_action_manager = actor_manager.architecture.architecture(torch.from_numpy(temp_obs).to(device)).detach()
        np.savetxt("ones_action_run_end.csv", temp_action_run.cpu().numpy(), delimiter=",")
        np.savetxt("ones_action_jump_end.csv", temp_action_jump.cpu().numpy(), delimiter=",")
        np.savetxt("ones_action_manager_end.csv", temp_action_manager.cpu().numpy(), delimiter=",")

    if update==1000 or update==3000 or update==5000 or update==7000 or update==9000:
        ppo.set_manager_training(False)

    if update==2000 or update==4000 or update==6000 or update==8000:
        ppo.set_manager_training(True)