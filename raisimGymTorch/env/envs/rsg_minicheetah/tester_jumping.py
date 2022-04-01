from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import rsg_minicheetah
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from torch.distributions import Categorical
from networkSelector import run_bool_function, run_bool_function_0, run_bool_function_1
import os
import math
import time
import raisimGymTorch.algo.ppo.module as ppo_module
import numpy as np
import torch
import argparse
import pygame
import csv
import datetime


# pygame for logitech gamepad
pygame.display.init()
pygame.joystick.init()
pygame.joystick.Joystick(0).init()


# configuration
parser = argparse.ArgumentParser()
parser.add_argument('-w', '--weight', help='trained weight path', type=str, default='')
args = parser.parse_args()

# directories
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../../../.."

# weight directory
weight_path_run = "../../../data/minicheetah_locomotion/baselineRun2/full_5000.pt"
iteration_number_run = weight_path_run.rsplit('/', 1)[1].split('_', 1)[1].rsplit('.', 1)[0]
weight_dir_run = weight_path_run.rsplit('/', 1)[0] + '/'

weight_path_jump = "../../../data/minicheetah_locomotion/baselineJump1-2/full_7500.pt"
iteration_number_jump = weight_path_jump.rsplit('/', 1)[1].split('_', 1)[1].rsplit('.', 1)[0]
weight_dir_jump = weight_path_jump.rsplit('/', 1)[0] + '/'

weight_path_manager = "../../../data/minicheetah_locomotion/2022-03-31-09-54-39/full_500.pt"
iteration_number_manager = weight_path_manager.rsplit('/', 1)[1].split('_', 1)[1].rsplit('.', 1)[0]
weight_dir_manager = weight_path_manager.rsplit('/', 1)[0] + '/'

# config
cfg = YAML().load(open(task_path + "/cfg.yaml", 'r')) # change to weight_path

# create environment from the configuration file
cfg['environment']['num_envs'] = 1

sensor_dim = 2
env = VecEnv(rsg_minicheetah.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)), cfg['environment'], sensor_dim=sensor_dim)

# shortcuts
ob_dim = env.num_obs
robotState_dim = env.num_robotState
act_dim = env.num_acts


if False:  # weight_path == "":
    print("Can't find trained weight, please provide a trained weight with --weight switch\n")
else:
    # print("Loaded weight from {}\n".format(weight_path))
    start = time.time()
    env.reset()
    reward_ll_sum = 0
    done_sum = 0
    average_dones = 0.
    n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
    total_steps = n_steps * 1
    start_step_id = 0

    print("Visualizing and evaluating the policy: ")
    actor_run = ppo_module.MLP(cfg['architecture']['policy_net'], torch.nn.LeakyReLU, ob_dim - sensor_dim + robotState_dim, act_dim)
    actor_jump = ppo_module.MLP(cfg['architecture']['policy_net'], torch.nn.LeakyReLU, ob_dim + robotState_dim, act_dim)
    actor_manager = ppo_module.MLP(cfg['architecture']['manager_net'], torch.nn.LeakyReLU, ob_dim + robotState_dim + 1, 2, softmax=True)
    actor_run.load_state_dict(torch.load(weight_path_run)['actor_architecture_state_dict'])
    actor_jump.load_state_dict(torch.load(weight_path_jump)['actor_architecture_state_dict'])
    actor_manager.load_state_dict(torch.load(weight_path_manager)['actor_architecture_state_dict'])  # actor_architecture_state_dict
    print('actor of {} parameters'.format(sum(p.numel() for p in actor_run.parameters())))
    print('actor of {} parameters'.format(sum(p.numel() for p in actor_jump.parameters())))

    estimator_run = ppo_module.MLP(cfg['architecture']['estimator_net'], torch.nn.LeakyReLU, ob_dim - sensor_dim, robotState_dim)
    estimator_jump = ppo_module.MLP(cfg['architecture']['estimator_net'], torch.nn.LeakyReLU, ob_dim - sensor_dim, robotState_dim)
    estimator_run.load_state_dict(torch.load(weight_path_run)['estimator_architecture_state_dict'])
    estimator_jump.load_state_dict(torch.load(weight_path_jump)['estimator_architecture_state_dict'])

    env.load_scaling(weight_dir_run, int(iteration_number_run), weight_dir_jump, int(iteration_number_jump),
                     weight_dir_manager, int(iteration_number_manager), one_directory=False)
    env.turn_on_visualization()
    env.start_video_recording(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "policy.mp4")
    time.sleep(2)

    # max_steps = 1000000
    max_steps = 400 ## 10 secs
    command = np.array([3.5, 0, 0], dtype=np.float32)
    env.set_command(command, testNumber=1)
    env.curriculum_callback(0)
    env.reset()
    selectedNetwork = None
    # env.printTest()

    run_bool = np.ones(shape=(cfg['environment']['num_envs'], 1), dtype=np.intc)
    selectionCalculation = -1 * np.ones(shape=(cfg['environment']['num_envs'], 1), dtype=np.intc)
    # run_bool = None

    for step in range(max_steps):
        frame_start = time.time()
        # if step % 400 == 0:
        #     command_Vx = np.random.uniform(-1.75, 3.5, 1)
        #     command_Vy = np.random.uniform(-1., 1., 1)
        #     command_yaw = np.random.uniform(-2., 2., 1)
        #     command = np.array([command_Vx, command_Vy, command_yaw], dtype=np.float32)
        #     env.set_command(command)

        [obs_run, obs_jump, obs_manager], obs_notNorm = env.observe(update_mean=False)
        obs_estimator_run = obs_run[:,:ob_dim-sensor_dim]
        obs_estimator_jump = obs_jump[:,:ob_dim-sensor_dim]
        robotState = env.getRobotState()
        est_out_run = estimator_run.architecture(torch.from_numpy(obs_estimator_run).cpu())
        est_out_jump = estimator_jump.architecture(torch.from_numpy(obs_estimator_jump).cpu())
        concatenated_obs_actor_run = np.concatenate((obs_run, est_out_run.cpu().detach().numpy()), axis=1)
        concatenated_obs_actor_jump = np.concatenate((obs_jump, est_out_jump.cpu().detach().numpy()), axis=1)
        concatenated_obs_actor_manager = np.concatenate((obs_manager, est_out_run.cpu().detach().numpy(), np.float32(run_bool)), axis=1)

        action_probs = actor_manager.architecture(torch.from_numpy(concatenated_obs_actor_manager).cpu())
        dist = Categorical(action_probs)
        bool_manager = dist.sample()
        run_bool = bool_manager.unsqueeze(1)
        # if step==0:
        #     run_bool = None
        # run_bool, selectionCalculation = run_bool_function(obs_notNorm, selectionCalculation, output=True, old_bool=run_bool)#only test!!!
        # run_bool = torch.from_numpy(run_bool)
        # run_bool = torch.from_numpy(run_bool_function_0(obs_notNorm)) #only test!!!
        # run_bool = torch.from_numpy(run_bool_function_1(obs_notNorm)) #only test!!!
        previousNetwork = selectedNetwork
        selectedNetwork = bool_manager[0].item()
        jump_bool = torch.add(torch.ones(run_bool.size(), device='cpu'), run_bool, alpha=-1)  # 1-run_bool
        actions_run = actor_run.architecture(torch.from_numpy(concatenated_obs_actor_run).cpu())
        actions_jump = actor_jump.architecture(torch.from_numpy(concatenated_obs_actor_jump).cpu())
        action_ll = run_bool * actions_run + jump_bool * actions_jump
        reward_ll, dones = env.step(action_ll.detach().numpy(), run_bool=run_bool.detach().numpy(), manager_training=True)
        # env.go_straight_controller()

        # f1 = open('randomCommandData.csv', 'a')
        # writer = csv.writer(f1)
        # writer.writerow([command[0][0], command[1][0], command[2][0]])
        #
        # f2 = open('velocityData.csv', 'a')
        # writer = csv.writer(f2)
        # writer.writerow([*robotState[0][0:2], obs[0][17]])
        #
        # f3 = open('estimatedVelocityData.csv', 'a')
        # writer = csv.writer(f3)
        # writer.writerow(est_in[0][0:2].cpu().detach().numpy())

        if step==0:
            if selectedNetwork == 0:
                print("selected network in step ", step, ": jump")
            else:
                print("selected network in step ", step, ": run")
        elif previousNetwork == 0 and selectedNetwork == 1:
            print("changed network in step ", step, ": jump -> run")
        elif previousNetwork == 1 and selectedNetwork == 0:
            print("changed network in step ", step, ": run -> jump")

        # if selectedNetwork == 0:
        #     print("jump selected")

        reward_ll_sum = reward_ll_sum + reward_ll[0]
        if dones or step == max_steps - 1:
            print('----------------------------------------------------')
            print('{:<40} {:>6}'.format("average ll reward: ", '{:0.10f}'.format(reward_ll_sum / (step + 1 - start_step_id))))
            print('{:<40} {:>6}'.format("time elapsed [sec]: ", '{:6.4f}'.format((step + 1 - start_step_id) * 0.01)))
            print('----------------------------------------------------\n')
            start_step_id = step + 1
            reward_ll_sum = 0.0

        frame_end = time.time()
        wait_time = cfg['environment']['control_dt'] - (frame_end-frame_start)
        if wait_time > 0.:
            time.sleep(wait_time)
        time.sleep(0.01) #0.05

    env.stop_video_recording()
    env.turn_off_visualization()
    env.reset()
    print("Finished at the maximum visualization steps")
