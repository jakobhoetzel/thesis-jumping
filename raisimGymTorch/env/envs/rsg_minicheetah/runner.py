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
import random

# from torch.distributions import Categorical
# m = Categorical(torch.tensor([ 0.25, 0.25, 0.25, 0.25 ]))
# ms = m.sample()  # equal probability of 0, 1, 2, 3
# msi = ms.item()
# exit()

# print("No training here")
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
if args.runNumber != "":
    runNumber = int(args.runNumber)
else:
    runNumber = None
#weight_path = args.weight
weight_path_run = "../../../data/minicheetah_locomotion/2022-04-14-22-53-54/full_3500.pt"
iteration_number_run = weight_path_run.rsplit('/', 1)[1].split('_', 1)[1].rsplit('.', 1)[0]
weight_dir_run = weight_path_run.rsplit('/', 1)[0] + '/'

weight_path_jump = "../../../data/minicheetah_locomotion/2022-04-14-09-33-21/full_5000.pt"
iteration_number_jump = weight_path_jump.rsplit('/', 1)[1].split('_', 1)[1].rsplit('.', 1)[0]
weight_dir_jump = weight_path_jump.rsplit('/', 1)[0] + '/'

# check if gpu is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# directories
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../../../.."

# config
cfg = YAML().load(open(task_path + "/cfg.yaml", 'r'))

# IL_lr = 5e-7
# PPO_lr = 5e-4
# if runNumber == 0:
#     learning_rate = 5e-5
# elif runNumber == 1:
#     learning_rate = 1e-6
# elif runNumber == 2:
#     learning_rate = 5e-6
learning_rate = 1e-7 #danach e-8

# create environment from the configuration file
sensor_dim = 2
env = VecEnv(rsg_minicheetah.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)), cfg['environment'], sensor_dim=sensor_dim)

seed = 1  # seed for reproducibility
random.seed(seed)
np.random.seed(abs(seed))
torch.manual_seed(seed)
env.seed(seed)

# shortcuts
ob_dim = env.num_obs  # including sensor
robotState_dim = env.num_robotState
act_dim = env.num_acts

# Training
n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])  # 400
total_steps = n_steps * env.num_envs  # 40000

avg_rewards = []

# savedWeights_run = torch.load(weight_path_run)['actor_architecture_state_dict']['architecture.0.weight']
# savedWeights_run_obs = savedWeights_run[:, :ob_dim-sensor_dim]  # no sensor data in standard running
# savedWeights_run_robotState = savedWeights_run[:, -robotState_dim:]
# addedWeights_run = torch.randn(512,sensor_dim,device=device)*0  # add nodes with low weights for sensor
# combinedWeights_run = torch.cat([savedWeights_run_obs, addedWeights_run, savedWeights_run_robotState], dim=1)
# savedDict_run = torch.load(weight_path_run)['actor_architecture_state_dict']
# savedDict_run['architecture.0.weight'] = torch.nn.Parameter(combinedWeights_run)
# mlpActor_run = ppo_module.MLP(cfg['architecture']['policy_net'], torch.nn.LeakyReLU, ob_dim + robotState_dim, act_dim).to(device)
# mlpActor_run.load_state_dict(savedDict_run)
# actor_run = ppo_module.Actor(mlpActor_run,
#                              ppo_module.MultivariateGaussianDiagonalCovariance(act_dim, 1.0),  # 1.0
#                              device)

mlpActor_run = ppo_module.MLP(cfg['architecture']['policy_net'], nn.LeakyReLU, ob_dim - sensor_dim + robotState_dim, act_dim).to(device)
mlpActor_run.load_state_dict(torch.load(weight_path_run)['actor_architecture_state_dict'])
actor_run = ppo_module.Actor(mlpActor_run,
                             ppo_module.MultivariateGaussianDiagonalCovariance(act_dim, 0.000001),  # 0.0 as not trained
                             device)

mlpActor_jump = ppo_module.MLP(cfg['architecture']['policy_net'], nn.LeakyReLU, ob_dim + robotState_dim, act_dim).to(device)
mlpActor_jump.load_state_dict(torch.load(weight_path_jump)['actor_architecture_state_dict'])
actor_jump = ppo_module.Actor(mlpActor_jump,
                             ppo_module.MultivariateGaussianDiagonalCovariance(act_dim, 0.000001),  # 0.0 as not trained
                             device)

# savedWeights_run = torch.load(weight_path_run)['actor_architecture_state_dict']['architecture.0.weight']
# savedDict_run = torch.load(weight_path_run)['actor_architecture_state_dict']
# savedDict_run['architecture.0.weight'] = torch.nn.Parameter(savedWeights_run)
# mlpActor_run = ppo_module.MLP(cfg['architecture']['policy_net'], torch.nn.LeakyReLU, ob_dim - sensor_dim + robotState_dim, act_dim).to(device)
# mlpActor_run.load_state_dict(savedDict_run)
# actor_run = ppo_module.Actor(mlpActor_run,
#                               ppo_module.MultivariateGaussianDiagonalCovariance(act_dim, 0.000001),  # 0.0 as not trained
#                               device)

# savedWeights_jump = torch.load(weight_path_jump)['actor_architecture_state_dict']['architecture.0.weight']
# savedDict_jump = torch.load(weight_path_jump)['actor_architecture_state_dict']
# savedDict_jump['architecture.0.weight'] = torch.nn.Parameter(savedWeights_jump)
# mlpActor_jump = ppo_module.MLP(cfg['architecture']['policy_net'], torch.nn.LeakyReLU, ob_dim + robotState_dim, act_dim).to(device)
# mlpActor_jump.load_state_dict(savedDict_jump)
# actor_jump = ppo_module.Actor(mlpActor_jump,
#                               ppo_module.MultivariateGaussianDiagonalCovariance(act_dim, 0.000001),  # 0.0 as not trained
#                               device)

mlpCritic_run = ppo_module.MLP(cfg['architecture']['value_net'], torch.nn.LeakyReLU, ob_dim + robotState_dim, 1)
mlpCritic_run.load_state_dict(torch.load(weight_path_run)['critic_architecture_state_dict'])
critic_run = ppo_module.Critic(mlpCritic_run, device)
mlpCritic_jump = ppo_module.MLP(cfg['architecture']['value_net'], torch.nn.LeakyReLU, ob_dim + robotState_dim, 1)
mlpCritic_jump.load_state_dict(torch.load(weight_path_jump)['critic_architecture_state_dict'])
critic_jump = ppo_module.Critic(mlpCritic_jump, device)

actor_manager = ppo_module.Manager(ppo_module.MLP(cfg['architecture']['manager_net'], nn.LeakyReLU, ob_dim + robotState_dim + 1, 2, softmax=True),  # +1 for former network
                                   device)
critic_manager = ppo_module.Critic(ppo_module.MLP(cfg['architecture']['value_net'], nn.LeakyReLU, ob_dim + robotState_dim + 1, 1),
                                device)
mlpEstimator_run = ppo_module.MLP(cfg['architecture']['estimator_net'], torch.nn.LeakyReLU, ob_dim - sensor_dim, robotState_dim).to(device)
mlpEstimator_run.load_state_dict(torch.load(weight_path_run)['estimator_architecture_state_dict'])
stateEstimator_run = ppo_module.StateEstimator(mlpEstimator_run,
                                           device)

mlpEstimator_jump = ppo_module.MLP(cfg['architecture']['estimator_net'], torch.nn.LeakyReLU, ob_dim - sensor_dim, robotState_dim).to(device)
mlpEstimator_jump.load_state_dict(torch.load(weight_path_jump)['estimator_architecture_state_dict'])
stateEstimator_jump = ppo_module.StateEstimator(mlpEstimator_jump,
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
              estimator_run=stateEstimator_run,
              estimator_jump=stateEstimator_jump,
              num_envs=cfg['environment']['num_envs'],
              num_transitions_per_env=n_steps,
              num_learning_epochs=4,
              clip_param=0.2,  # 0.2
              gamma=0.95,  # 0.99
              lam=0.95,
              learning_rate=learning_rate,  # 5e-4
              learning_rate_critic=1e-4,  # 5e-4
              sl_learning_rate=5e-6,
              entropy_coef=0.01,
              num_mini_batches=8,
              device=device,
              log_dir=saver.data_dir,
              shuffle_batch=False,
              # manager_training=False,
              )

data_tags = env.get_step_data_tag()

scheduler = torch.optim.lr_scheduler.MultiStepLR(ppo.optimizer, milestones=[2000], gamma=0.333333)

# if mode == 'retrain':
#     load_param(weight_path, env, actor_run, actor_jump, critic_run, critic_jump, stateEstimator, ppo.optimizer, saver.data_dir)

max_iteration = 7500 + 1  # 5000+1

env.load_scaling(weight_dir_run, int(iteration_number_run), weight_dir_jump, int(iteration_number_jump), weight_dir_jump, int(iteration_number_jump),
                 1e8, one_directory=False)  # 1e8 -> less disruption when retraining  # use jump scaling as it includes sensor data
                                                                                # for normalisation of observation

# ppo.set_manager_training(True)
# ppo.set_manager_training(False)  # train run and jump network
# if "IL_end" not in locals():
#     IL_end = 25
actorManagerUpdate = False  # only update critic
# freeze_manager(ppo)
# freeze_actors(ppo)
testNumber = 0 # 0=both, 1=only jumping, 2=without hurdle
# onlyRunTrainingEveryN = 9

for update in range(max_iteration):
    start = time.time()
    # if update == 0:
    #     testNumber = 1
    #     env.set_command(np.array([3.5, 0, 0], dtype=np.float32), testNumber=testNumber)  # to only train in jumping environment
    # if update == IL_end+1:
    #     testNumber = 0
    #     env.set_command(np.array([0, 0, 0], dtype=np.float32), testNumber=testNumber)  # to train both environments
    # if (update > 0) and (update < IL_end+1) and (update % onlyRunTrainingEveryN == 0):
    #     testNumber = 2  # without hurdles
    #     env.set_command(np.array([0, 0, 0], dtype=np.float32), testNumber=testNumber)
    # elif (update > 0) and (update < IL_end+1):
    #     testNumber = 1  # always hurdles
    #     env.set_command(np.array([3.5, 0, 0], dtype=np.float32), testNumber=testNumber)

    env.reset()
    reward_ll_sum = 0
    done_sum = 0
    average_dones = 0.
    run_bool = np.ones(shape=(cfg['environment']['num_envs'], 1), dtype=np.intc)
    selectionCalculation = -1 * np.ones(shape=(cfg['environment']['num_envs'], 1), dtype=np.intc)

    env.curriculum_callback(update)  # start with half height

    if update % cfg['environment']['eval_every_n'] == 0:
        print("Visualizing and evaluating the current policy")
        torch.save({
            'actor_run_architecture_state_dict': actor_run.architecture.state_dict(),
            'actor_run_distribution_state_dict': actor_run.distribution.state_dict(),
            'actor_jump_architecture_state_dict': actor_jump.architecture.state_dict(),
            'actor_jump_distribution_state_dict': actor_jump.distribution.state_dict(),
            # 'actor_manager_architecture_state_dict': actor_manager.architecture.state_dict(),
            'critic_run_architecture_state_dict': critic_run.architecture.state_dict(),
            'critic_jump_architecture_state_dict': critic_jump.architecture.state_dict(),
            'estimator_run_architecture_state_dict': stateEstimator_run.architecture.state_dict(),
            'estimator_jump_architecture_state_dict': stateEstimator_jump.architecture.state_dict(),
            # 'actor_architecture_state_dict': actor_manager.architecture.state_dict(),
            # 'critic_architecture_state_dict': critic_manager.architecture.state_dict(),
            'optimizer_state_dict': ppo.optimizer.state_dict(),
        }, saver.data_dir+"/full_"+str(update)+'.pt')
        actor_run.save_deterministic_graph(saver.data_dir + "/actor_run_" + str(update) + ".pt", torch.rand(1, ob_dim - sensor_dim + robotState_dim).cpu())
        actor_jump.save_deterministic_graph(saver.data_dir + "/actor_jump_" + str(update) + ".pt", torch.rand(1, ob_dim + robotState_dim).cpu())
        # actor_manager.save_deterministic_graph(saver.data_dir + "/actor_manager_" + str(update) + ".pt", torch.rand(1, ob_dim + robotState_dim).cpu())
        # actor_manager.save_deterministic_graph(saver.data_dir + "/actor_" + str(update) + ".pt", torch.rand(1, ob_dim + robotState_dim + 1).cpu())
        stateEstimator_run.save_deterministic_graph(saver.data_dir + "/estimator_run_" + str(update) + ".pt", torch.rand(1, ob_dim-sensor_dim).cpu())
        stateEstimator_jump.save_deterministic_graph(saver.data_dir + "/estimator_jump_" + str(update) + ".pt", torch.rand(1, ob_dim-sensor_dim).cpu())
        critic_run.save_deterministic_graph(saver.data_dir + "/critic_run_" + str(update) + ".pt", torch.rand(1, ob_dim + robotState_dim).cpu())
        critic_jump.save_deterministic_graph(saver.data_dir + "/critic_jump_" + str(update) + ".pt", torch.rand(1, ob_dim + robotState_dim).cpu())

        # temp_obs = np.ones((cfg['environment']['num_envs'], ob_dim + robotState_dim), dtype=np.float32)  # to see if input network changes
        # temp_action_run = actor_run.architecture.architecture(torch.from_numpy(temp_obs).to(device)).detach()
        # temp_action_jump = actor_jump.architecture.architecture(torch.from_numpy(temp_obs).to(device)).detach()
        # temp_action_manager = actor_manager.architecture.architecture(torch.from_numpy(temp_obs).to(device)).detach()
        # np.savetxt("ones_action_run_beg.csv", temp_action_run.cpu().numpy(), delimiter=",")
        # np.savetxt("ones_action_jump_beg.csv", temp_action_jump.cpu().numpy(), delimiter=",")
        # np.savetxt("ones_action_manager_beg.csv", temp_action_manager.cpu().numpy(), delimiter=",")

        # we create another graph just to demonstrate the save/load method
        # loaded_graph = ppo_module.MLP(cfg['architecture']['policy_net'], torch.nn.LeakyReLU, ob_dim + robotState_dim, act_dim)
        # loaded_graph.load_state_dict(torch.load(saver.data_dir+"/full_"+str(update)+'.pt')['actor_architecture_state_dict'])

        env.turn_on_visualization()
        env.start_video_recording(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "policy_"+str(update)+'.mp4')
        time.sleep(2) #1

        for step in range(n_steps*1):  # n_steps*2
            frame_start = time.time()
            [obs_run, obs_jump, obs_manager], obs_notNorm = env.observe(update_mean=False, update_manager=ppo.manager_training)  # don't update mean and var
            obs_noSensor_run = obs_run[:,:ob_dim-sensor_dim]
            obs_noSensor_jump = obs_jump[:,:ob_dim-sensor_dim]
            robotState = env.getRobotState()
            est_out_run = stateEstimator_run.predict(torch.from_numpy(obs_noSensor_run).to(device))
            est_out_jump = stateEstimator_jump.predict(torch.from_numpy(obs_noSensor_jump).to(device))
            concatenated_obs_actor_run = np.concatenate((obs_noSensor_run, est_out_run.cpu().detach().numpy()), axis=1)
            concatenated_obs_critic_run = np.concatenate((obs_run, est_out_run.cpu().detach().numpy()), axis=1) #in case of run different
            concatenated_obs_jump = np.concatenate((obs_jump, est_out_jump.cpu().detach().numpy()), axis=1)
            action, run_bool = ppo.observe(concatenated_obs_actor_run, concatenated_obs_critic_run, concatenated_obs_jump)
            # if update <= IL_end:
            #     if step ==0:
            #         old_bool=None
            #     run_bool, selectionCalculation = run_bool_function(obs_notNorm, selectionCalculation, output=False, old_bool=old_bool)
            #     action, _ = ppo.observe(concatenated_obs_actor_run, concatenated_obs_actor_jump, concatenated_obs_actor_manager, run_bool)
            #     old_bool = run_bool

            # action_ll, _ = actor.sample(torch.from_numpy(concatenated_obs_actor).to(device))  # stochastic action
            # action_ll = loaded_graph.architecture(torch.from_numpy(obs).cpu())

            reward_ll, dones = env.step(action, run_bool, ppo.manager_training)  # in stochastic action case
            # env.go_straight_controller()
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
    run_bool = np.ones(shape=(cfg['environment']['num_envs'], 1), dtype=np.float32)
    loss_IL_sum = 0
    selectionCalculation = -1 * np.ones(shape=(cfg['environment']['num_envs'], 1), dtype=np.intc)

# actual training
    for step in range(n_steps):
        [obs_run, obs_jump, obs_manager], obs_notNorm = env.observe(update_manager=ppo.manager_training)  # observation differ due to normalisation
        obs_noSensor_run = obs_run[:,:ob_dim-sensor_dim]
        obs_noSensor_jump = obs_jump[:,:ob_dim-sensor_dim]
        robotState = env.getRobotState()
        est_out_run = stateEstimator_run.predict(torch.from_numpy(obs_noSensor_run).to(device))
        est_out_jump = stateEstimator_jump.predict(torch.from_numpy(obs_noSensor_jump).to(device))
        concatenated_obs_actor_run = np.concatenate((obs_noSensor_run, est_out_run.cpu().detach().numpy()), axis=1)
        concatenated_obs_critic_run = np.concatenate((obs_run, est_out_run.cpu().detach().numpy()), axis=1) #in case of run different
        concatenated_obs_jump = np.concatenate((obs_jump, est_out_jump.cpu().detach().numpy()), axis=1)
        action, run_bool = ppo.observe(concatenated_obs_actor_run, concatenated_obs_critic_run, concatenated_obs_jump)

        # if update <= IL_end:
        #     run_bool, selectionCalculation = run_bool_function(obs_notNorm, selectionCalculation, output=False, old_bool=old_bool, test_number=testNumber)
        #     action, _ = ppo.observe(concatenated_obs_actor_run, concatenated_obs_actor_jump, concatenated_obs_actor_manager, run_bool)
        #     # loss_IL_sum += IL.identity_learning(actor_manager=actor_manager, guideline=run_bool, obs=concatenated_obs_actor_manager, device=device, lr=IL_lr)

        reward, dones = env.step(action, run_bool, ppo.manager_training)
        # env.go_straight_controller()
        ppo.step(value_obs_run=concatenated_obs_critic_run, value_obs_jump=concatenated_obs_jump, value_obs_manager=None,
                 est_obs=None, robotState=robotState, rews=reward, dones=dones, guideline=run_bool)
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
    [obs_run, obs_jump, obs_manager], obs_notNorm = env.observe(update_manager=ppo.manager_training)
    obs_noSensor_run = obs_run[:,:ob_dim-sensor_dim]
    obs_noSensor_jump = obs_jump[:,:ob_dim-sensor_dim]
    robotState = env.getRobotState()
    est_out_run = stateEstimator_run.predict(torch.from_numpy(obs_noSensor_run).to(device))
    est_out_jump = stateEstimator_jump.predict(torch.from_numpy(obs_noSensor_jump).to(device))
    concatenated_obs_actor_run = np.concatenate((obs_noSensor_run, est_out_run.cpu().detach().numpy()), axis=1)
    concatenated_obs_critic_run = np.concatenate((obs_run, est_out_run.cpu().detach().numpy()), axis=1) #in case of run different
    concatenated_obs_jump = np.concatenate((obs_jump, est_out_jump.cpu().detach().numpy()), axis=1)
    action, run_bool = ppo.observe(concatenated_obs_actor_run, concatenated_obs_critic_run, concatenated_obs_jump)

    # a0 = list(actor_run.parameters())[0].grad
    # print("a0: ", a0)
    # a1 = list(actor_jump.parameters())[0].grad
    # print("a1: ", a1)
    # a2 = list(actor_manager.parameters())[0].grad
    # print("a2: ", a2)

    # if update <= IL_end:
    #     loss_IL_sum += ppo.supervised_learning()
    # if update > IL_end:
    #     actorManagerUpdate = True
    ppo.update(actor_obs_manager=None, value_obs_run=concatenated_obs_critic_run,
                   value_obs_jump=concatenated_obs_jump, value_obs_manager=None,
                   log_this_iteration=update % 20 == 0, update=update, actor_manager_update=actorManagerUpdate)

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
    # print('std: ')
    # print(np.exp(actor_run.distribution.std.cpu().detach().numpy()))
    # print(np.exp(actor_jump.distribution.std.cpu().detach().numpy()))
    # print('----------------------------------------------------\n')

    # if update <= IL_end:
    #     print('loss SL: ', loss_IL_sum)
    #     print('----------------------------------------------------\n')


    # if update % cfg['environment']['eval_every_n'] == 0:
    #     temp_obs = np.ones((cfg['environment']['num_envs'], ob_dim + robotState_dim), dtype=np.float32)  # to see if input network changes
    #     temp_action_run = actor_run.architecture.architecture(torch.from_numpy(temp_obs).to(device)).detach()
    #     temp_action_jump = actor_jump.architecture.architecture(torch.from_numpy(temp_obs).to(device)).detach()
    #     temp_action_manager = actor_manager.architecture.architecture(torch.from_numpy(temp_obs).to(device)).detach()
    #     np.savetxt("ones_action_run_end.csv", temp_action_run.cpu().numpy(), delimiter=",")
    #     np.savetxt("ones_action_jump_end.csv", temp_action_jump.cpu().numpy(), delimiter=",")
    #     np.savetxt("ones_action_manager_end.csv", temp_action_manager.cpu().numpy(), delimiter=",")

    # if update==1000 or update==3000 or update==5000 or update==7000 or update==9000:
    #     ppo.set_manager_training(True)
    #
    # if update==2000 or update==4000 or update==6000 or update==8000:
    #     ppo.set_manager_training(False)