from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import rsg_minicheetah
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from torch.distributions import Categorical
from networkSelector import NetworkSelector
from gradientCalculation import gradient_calculation
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
import random


# # pygame for logitech gamepad
# pygame.display.init()
# pygame.joystick.init()
# pygame.joystick.Joystick(1).init()


# configuration
parser = argparse.ArgumentParser()
parser.add_argument('-w', '--weight', help='trained weight path', type=str, default='')
args = parser.parse_args()
device = 'cpu'

# directories
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../../../.."

# weight directory
weight_path_run = None; weight_path_jump = None; weight_path_manager = None; weight_path_total = None
iteration_number_run = None; iteration_number_jump = None; iteration_number_manager = None; iteration_number_total = None
weight_dir_run = None; weight_dir_jump = None; weight_dir_manager = None; weight_dir_total = None

weight_path_run = "../../../data/minicheetah_locomotion/baselineRun_Switch1_Critic/full_0.pt"
iteration_number_run = weight_path_run.rsplit('/', 1)[1].split('_', 1)[1].rsplit('.', 1)[0]
weight_dir_run = weight_path_run.rsplit('/', 1)[0] + '/'

weight_path_jump = "../../../data/minicheetah_locomotion/baselineJump1-6/full_7500.pt"
# weight_path_jump = "../../../data/minicheetah_locomotion/2022-07-13-04-21-41/full_7500.pt"
iteration_number_jump = weight_path_jump.rsplit('/', 1)[1].split('_', 1)[1].rsplit('.', 1)[0]
weight_dir_jump = weight_path_jump.rsplit('/', 1)[0] + '/'

weight_path_manager = "../../../data/minicheetah_locomotion/2022-04-06-11-07-33/full_25.pt"
iteration_number_manager = weight_path_manager.rsplit('/', 1)[1].split('_', 1)[1].rsplit('.', 1)[0]
weight_dir_manager = weight_path_manager.rsplit('/', 1)[0] + '/'

# weight_path_total = "../../../data/minicheetah_locomotion/2022-04-54-16-53-54/full_7000.pt"
# iteration_number_total = weight_path_total.rsplit('/', 1)[1].split('_', 1)[1].rsplit('.', 1)[0]
# weight_dir_total = weight_path_total.rsplit('/', 1)[0] + '/'

# config
cfg = YAML().load(open(task_path + "/cfg.yaml", 'r')) # change to weight_path

# create environment from the configuration file
cfg['environment']['num_envs'] = 1  # 1 to see

sensor_dim = 2
env = VecEnv(rsg_minicheetah.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)), cfg['environment'], sensor_dim=sensor_dim)
networkSelector = NetworkSelector(cfg['environment']['num_envs'], device)

seed = random.randint(-10000, 10000)  # seed for reproducibility
# random.seed(seed)
np.random.seed(abs(seed))
torch.manual_seed(seed)
env.seed(seed)

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

    oneDirectory = False
    print("Visualizing and evaluating the policy: ")
    if oneDirectory:
        actor_run = ppo_module.MLP(cfg['architecture']['policy_net'], torch.nn.LeakyReLU,
                                   ob_dim - sensor_dim + robotState_dim, act_dim)
        actor_jump = ppo_module.MLP(cfg['architecture']['policy_net'], torch.nn.LeakyReLU, ob_dim + robotState_dim,
                                    act_dim)
        critic_run = ppo_module.MLP(cfg['architecture']['value_net'], torch.nn.LeakyReLU, ob_dim + robotState_dim, 1)
        critic_jump = ppo_module.MLP(cfg['architecture']['value_net'], torch.nn.LeakyReLU, ob_dim + robotState_dim, 1)
        # actor_manager = ppo_module.MLP(cfg['architecture']['manager_net'], torch.nn.LeakyReLU, ob_dim + robotState_dim + 1, 2, softmax=True)
        actor_run.load_state_dict(torch.load(weight_path_total)['actor_run_architecture_state_dict'])
        actor_jump.load_state_dict(torch.load(weight_path_total)['actor_jump_architecture_state_dict'])
        critic_run.load_state_dict(torch.load(weight_path_total)['critic_run_architecture_state_dict'])
        critic_jump.load_state_dict(torch.load(weight_path_total)['critic_jump_architecture_state_dict'])
        # actor_manager.load_state_dict(torch.load(weight_path_manager)['actor_architecture_state_dict'])  # actor_architecture_state_dict
        estimator_run = ppo_module.MLP(cfg['architecture']['estimator_net'], torch.nn.LeakyReLU, ob_dim - sensor_dim,
                                       robotState_dim)
        estimator_jump = ppo_module.MLP(cfg['architecture']['estimator_net'], torch.nn.LeakyReLU, ob_dim - sensor_dim,
                                        robotState_dim)
        estimator_run.load_state_dict(torch.load(weight_path_total)['estimator_run_architecture_state_dict'])
        estimator_jump.load_state_dict(torch.load(weight_path_total)['estimator_jump_architecture_state_dict'])

        env.load_scaling(weight_dir_total, int(iteration_number_total), weight_dir_total, int(iteration_number_total),
                         weight_dir_total, int(iteration_number_total), one_directory=True)
    else:
        actor_run = ppo_module.MLP(cfg['architecture']['policy_net'], torch.nn.LeakyReLU,
                                   ob_dim - sensor_dim + robotState_dim, act_dim)
        actor_jump = ppo_module.MLP(cfg['architecture']['policy_net'], torch.nn.LeakyReLU, ob_dim + robotState_dim,
                                    act_dim)
        critic_run = ppo_module.MLP(cfg['architecture']['value_net'], torch.nn.LeakyReLU, ob_dim + robotState_dim, 1)
        critic_jump = ppo_module.MLP(cfg['architecture']['value_net'], torch.nn.LeakyReLU, ob_dim + robotState_dim, 1)
        # actor_manager = ppo_module.MLP(cfg['architecture']['manager_net'], torch.nn.LeakyReLU, ob_dim + robotState_dim + 1, 2, softmax=True)
        actor_run.load_state_dict(torch.load(weight_path_run)['actor_architecture_state_dict'])
        actor_jump.load_state_dict(torch.load(weight_path_jump)['actor_architecture_state_dict'])
        critic_run.load_state_dict(torch.load(weight_path_run)['critic_architecture_state_dict'])
        critic_jump.load_state_dict(torch.load(weight_path_jump)['critic_architecture_state_dict'])
        # actor_manager.load_state_dict(torch.load(weight_path_manager)['actor_architecture_state_dict'])  # actor_architecture_state_dict
        estimator_run = ppo_module.MLP(cfg['architecture']['estimator_net'], torch.nn.LeakyReLU, ob_dim - sensor_dim,
                                       robotState_dim)
        estimator_jump = ppo_module.MLP(cfg['architecture']['estimator_net'], torch.nn.LeakyReLU, ob_dim - sensor_dim,
                                        robotState_dim)
        estimator_run.load_state_dict(torch.load(weight_path_run)['estimator_architecture_state_dict'])
        estimator_jump.load_state_dict(torch.load(weight_path_jump)['estimator_architecture_state_dict'])

        env.load_scaling(weight_dir_run, int(iteration_number_run), weight_dir_jump, int(iteration_number_jump),
                         weight_dir_manager, int(iteration_number_manager), one_directory=False)

    print('actor of {} parameters'.format(sum(p.numel() for p in actor_run.parameters())))
    print('actor of {} parameters'.format(sum(p.numel() for p in actor_jump.parameters())))

    # env.turn_on_visualization()
    # env.start_video_recording(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "policy.mp4")
    time.sleep(2)

    # max_steps = 1000000
    max_steps = 350 ## 400*2 10 secs
    # command = np.array([random.uniform(3.0, 3.5), 0, 0], dtype=np.float32)
    command = np.array([3.5, 0, 0], dtype=np.float32)
    env.set_command(command, testNumber=1)
    env.curriculum_callback(0)
    env.reset()
    selectedNetwork = None
    # env.printTest()
    projectDistance = False
    dist_project = 0.0
    timestep = cfg['environment']['control_dt']
    getEveryNthStep = 200  # from environment
    sim_steps_per_control = round(cfg['environment']['control_dt']/cfg['environment']['simulation_dt']/getEveryNthStep)

    run_bool = np.ones(shape=(cfg['environment']['num_envs'], 1), dtype=np.intc)
    dones = np.zeros(shape=(cfg['environment']['num_envs'], 1), dtype=np.intc)
    networkSelectionSave = np.zeros((max_steps*sim_steps_per_control, 1))
    concatenated_obs_actor_jump = None
    concatenated_obs_critic_run = None
    # run_bool = None

    for step in range(max_steps):
        frame_start = time.time()

        # if (step % 10 == 0):
        #     command_Vx = -4. * pygame.joystick.Joystick(1).get_axis(1)
        #     if (command_Vx < 0):
        #         command_Vx *= 0.5
        #     command_Vy = - pygame.joystick.Joystick(1).get_axis(0)
        #     command_yaw = -2 * pygame.joystick.Joystick(1).get_axis(3)
        #     command = np.array([command_Vx, command_Vy, command_yaw], dtype=np.float32)
        #     env.set_command(command, testNumber=1)
        #     # command_change = - pygame.joystick.Joystick(1).get_axis(2)  # LT
        # if step % 400 == 0:
        #     command_Vx = np.random.uniform(-1.75, 3.5, 1)
        #     command_Vy = np.random.uniform(-1., 1., 1)
        #     command_yaw = np.random.uniform(-2., 2., 1)
        #     command = np.array([command_Vx, command_Vy, command_yaw], dtype=np.float32)
        #     env.set_command(command, testNumber=1)

        if concatenated_obs_actor_jump is not None:
            concatenated_obs_actor_jump_old = concatenated_obs_actor_jump.copy() #for gradient calculation
            concatenated_obs_critic_run_old = concatenated_obs_critic_run.copy()
        else:
            concatenated_obs_actor_jump_old = None
            concatenated_obs_critic_run_old = None

        [obs_run, obs_jump, obs_manager], obs_notNorm = env.observe(update_mean=False)
        obs_noSensor_run = obs_run[:,:ob_dim-sensor_dim]
        obs_noSensor_jump = obs_jump[:,:ob_dim-sensor_dim]
        robotState = env.getRobotState()
        est_out_run = estimator_run.architecture(torch.from_numpy(obs_noSensor_run).cpu())
        est_out_jump = estimator_jump.architecture(torch.from_numpy(obs_noSensor_jump).cpu())
        concatenated_obs_actor_run = np.concatenate((obs_noSensor_run, est_out_run.cpu().detach().numpy()), axis=1)
        concatenated_obs_critic_run = np.concatenate((obs_run, est_out_run.cpu().detach().numpy()), axis=1) #in case of run different
        concatenated_obs_actor_jump = np.concatenate((obs_jump, est_out_jump.cpu().detach().numpy()), axis=1)
        # concatenated_obs_actor_manager = np.concatenate((obs_manager, est_out_run.cpu().detach().numpy(), np.float32(run_bool)), axis=1)

        # project distance based on velocity when close to hurdle
        if False:
            dist_real = obs_notNorm.item(0,-1)
            # if 0.5 > dist_real > 0.0 and not projectDistance:
            if not run_bool[0,0] and not projectDistance and not dist_project == 5:
                projectDistance = True
                dist_project = dist_real + random.uniform(-0.1, 0.1)
            elif projectDistance:
                dist_project = dist_project - est_out_jump.data.numpy().item(0,0) * timestep
                if dist_project < -0.3:
                    dist_project = 5
                    projectDistance = False
                dist_project_norm = (dist_project - env.obs_rms_jump.mean[0,-1]) / np.sqrt(env.obs_rms_jump.var[0,-1]) # normalisation
                concatenated_obs_actor_jump[0,ob_dim-1] = dist_project_norm  # normalisation missing
                obs_notNorm[0,ob_dim-1] = dist_project
                print("real: ", dist_real, ", projected: ", dist_project)

            if run_bool[0,0] and dist_project == 5:  # after switch back to run
                dist_project = 0.0



        # action_probs = actor_manager.architecture(torch.from_numpy(concatenated_obs_actor_manager).cpu())
        # print(action_probs)
        # dist = Categorical(action_probs)
        # bool_manager = dist.sample()
        # run_bool = bool_manager.unsqueeze(1)
        value_run = critic_run.architecture(torch.from_numpy(concatenated_obs_critic_run).cpu())
        value_jump = critic_jump.architecture(torch.from_numpy(concatenated_obs_actor_jump).cpu())

        # gradient_calculation(critic_run, concatenated_obs_critic_run, concatenated_obs_critic_run_old)
        # gradient_calculation(critic_jump, concatenated_obs_actor_jump, concatenated_obs_actor_jump_old)

        # print('value run: ', value_run.item(), '    value_jump: ', value_jump.item())
        # if command_change > 0:
        #     run_bool = networkSelector.run_bool_function(value_run, value_jump, dones, obs_notNorm, 5)
        # else:
        #     run_bool = networkSelector.run_bool_function(value_run, value_jump, dones, obs_notNorm, 4)
        run_bool = networkSelector.run_bool_function(value_run, value_jump, dones, obs_notNorm, 3)
        # run_bool = networkSelector.run_bool_function(value_run, value_jump, dones, obs_notNorm, 5)
                                    # 0=pure value, 1=smoothing, 2=change after steps, 3=manual on dist, 4=run, 5=jump
        networkSelectionSave[step*sim_steps_per_control:(step+1)*sim_steps_per_control] = run_bool[0]*np.ones((sim_steps_per_control, 1))
        # run_bool = value_run > value_jump
        # run_bool = torch.from_numpy(run_bool_function_0(obs_notNorm)) #only test!!!
        # run_bool = torch.from_numpy(run_bool_function_1(obs_notNorm)) #only test!!!
        # previousNetwork = selectedNetwork
        # selectedNetwork = run_bool.item()
        jump_bool = torch.add(torch.ones(run_bool.size(), device='cpu'), run_bool, alpha=-1)  # 1-run_bool
        actions_run = actor_run.architecture(torch.from_numpy(concatenated_obs_actor_run).cpu())
        actions_jump = actor_jump.architecture(torch.from_numpy(concatenated_obs_actor_jump).cpu())
        action_ll = run_bool * actions_run + jump_bool * actions_jump
        reward_ll, dones = env.step(action_ll.detach().numpy(), run_bool=run_bool.detach().numpy().astype(bool), manager_training=False)
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

        # if step==0:
        #     if selectedNetwork == 0:
        #         print("selected network in step ", step, ": jump")
        #     else:
        #         print("selected network in step ", step, ": run")
        # elif previousNetwork == 0 and selectedNetwork == 1:
        #     print("changed network in step ", step, ": jump -> run")
        # elif previousNetwork == 1 and selectedNetwork == 0:
        #     print("changed network in step ", step, ": run -> jump")

        # if selectedNetwork == 0:
        #     print("jump selected")

        reward_ll_sum = reward_ll_sum + reward_ll[0]
        # if dones or step == max_steps - 1:
        #     print('----------------------------------------------------')
        #     print('{:<40} {:>6}'.format("average ll reward: ", '{:0.10f}'.format(reward_ll_sum / (step + 1 - start_step_id))))
        #     print('{:<40} {:>6}'.format("time elapsed [sec]: ", '{:6.4f}'.format((step + 1 - start_step_id) * 0.01)))
        #     print('----------------------------------------------------\n')
        #     start_step_id = step + 1
        #     reward_ll_sum = 0.0

        frame_end = time.time()
        wait_time = cfg['environment']['control_dt'] - (frame_end-frame_start)
        if wait_time > 0.:
            time.sleep(wait_time)
        time.sleep(0.01) #0.05 ONLY TO SEE, NOT FOR REAL SPEED!

    # env.stop_video_recording()
    # env.turn_off_visualization()
    # env.reset()

    # runInfo = np.concatenate((env.get_run_information()[:,1:], networkSelectionSave.transpose()),0)
    # runInfoOld = np.loadtxt("runInformation.csv", delimiter=",")
    # np.savetxt("runInformation.csv", runInfo.transpose(), delimiter=",")
    print("Finished at the maximum visualization steps")
