from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import rsg_minicheetah
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from networkSelector import run_bool_function
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
weight_path = args.weight
iteration_number = weight_path.rsplit('/', 1)[1].split('_', 1)[1].rsplit('.', 1)[0]
weight_dir = weight_path.rsplit('/', 1)[0] + '/'

# config
cfg = YAML().load(open(task_path + "/cfg.yaml", 'r')) # change to weight_path

# create environment from the configuration file
cfg['environment']['num_envs'] = 1

env = VecEnv(rsg_minicheetah.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)), cfg['environment'])

# shortcuts
ob_dim = env.num_obs
robotState_dim = env.num_robotState
act_dim = env.num_acts
sensor_dim = 3


if weight_path == "":
    print("Can't find trained weight, please provide a trained weight with --weight switch\n")
else:
    print("Loaded weight from {}\n".format(weight_path))
    start = time.time()
    env.reset()
    reward_ll_sum = 0
    done_sum = 0
    average_dones = 0.
    n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
    total_steps = n_steps * 1
    start_step_id = 0

    print("Visualizing and evaluating the policy: ", weight_path)
    actor_run = ppo_module.MLP(cfg['architecture']['policy_net'], torch.nn.LeakyReLU, ob_dim + robotState_dim, act_dim)
    actor_jump = ppo_module.MLP(cfg['architecture']['policy_net'], torch.nn.LeakyReLU, ob_dim + robotState_dim, act_dim)
    actor_run.load_state_dict(torch.load(weight_path)['actor_run_architecture_state_dict'])
    actor_jump.load_state_dict(torch.load(weight_path)['actor_jump_architecture_state_dict'])
    print('actor of {} parameters'.format(sum(p.numel() for p in actor_run.parameters())))
    print('actor of {} parameters'.format(sum(p.numel() for p in actor_jump.parameters())))

    estimator = ppo_module.MLP(cfg['architecture']['estimator_net'], torch.nn.LeakyReLU, ob_dim - sensor_dim, robotState_dim)
    estimator.load_state_dict(torch.load(weight_path)['estimator_architecture_state_dict'])

    env.load_scaling(weight_dir, int(iteration_number))
    env.turn_on_visualization()
    env.start_video_recording(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "policy.mp4")
    time.sleep(2)

    max_steps = 1000000
    ##max_steps = 400 ## 10 secs
    command = np.array([4.0, 0, 0], dtype=np.float32)
    env.set_command(command)
    env.curriculum_callback(5000)
    env.reset()
    env.printTest()

    run_bool = np.ones((1,1))

    for step in range(max_steps):
        frame_start = time.time()
        # if step % 400 == 0:
        #     command_Vx = np.random.uniform(-1.75, 3.5, 1)
        #     command_Vy = np.random.uniform(-1., 1., 1)
        #     command_yaw = np.random.uniform(-2., 2., 1)
        #     command = np.array([command_Vx, command_Vy, command_yaw], dtype=np.float32)
        #     env.set_command(command)

        obs = env.observe(False)
        obs_estimator = obs[:,:ob_dim-sensor_dim]
        robotState = env.getRobotState()
        # robotState = np.ones((cfg['environment']['num_envs'],robotState_dim), dtype=np.float32)  # for checking
        est_out = estimator.architecture(torch.from_numpy(obs_estimator).cpu())
        concatenated_obs_actor = np.concatenate((obs, est_out.cpu().detach().numpy()), axis=1)
        run_bool = torch.from_numpy(run_bool_function(obs, act_dim, True, run_bool)).cpu()
        jump_bool = torch.add(torch.ones(run_bool.size()).cpu(), run_bool, alpha=-1)
        action_ll = run_bool * actor_run.architecture(torch.from_numpy(concatenated_obs_actor).cpu())\
                    + jump_bool * actor_jump.architecture(torch.from_numpy(concatenated_obs_actor).cpu())
        reward_ll, dones = env.step(action_ll.cpu().detach().numpy())
        env.go_straight_controller()

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
        time.sleep(5)
        if wait_time > 0.:
            time.sleep(wait_time)

    env.stop_video_recording()
    env.turn_off_visualization()
    env.reset()
    print("Finished at the maximum visualization steps")
