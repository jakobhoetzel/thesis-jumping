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

# check if gpu is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# directories
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../../../.."

# whenever changing the saved file, modify update_number and weight_path
update_number = 0

weight_path = os.path.dirname(os.path.realpath(__file__)) + "/../../../data/minicheetah_locomotion/2021-02-09-23-44-33"
checkpoint_weight_path = weight_path + "/full_" + str(update_number) + ".pt"
print("Weight path: ", checkpoint_weight_path)

# config
cfg = YAML().load(open(weight_path + "/cfg.yaml", 'r'))

# create environment from the configuration file
env = VecEnv(rsg_minicheetah.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)), cfg['environment'])

# shortcuts
ob_dim = env.num_obs
act_dim = env.num_acts

# Training
n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])  # 400

actor = ppo_module.Actor(ppo_module.MLP(cfg['architecture']['policy_net'], nn.LeakyReLU, ob_dim, act_dim),
                         ppo_module.MultivariateGaussianDiagonalCovariance(act_dim, 1.0),
                         device)
critic = ppo_module.Critic(ppo_module.MLP(cfg['architecture']['value_net'], nn.LeakyReLU, ob_dim, 1),
                           device)

ppo = PPO.PPO(actor=actor,
              critic=critic,
              num_envs=cfg['environment']['num_envs'],
              num_transitions_per_env=n_steps,
              num_learning_epochs=4,
              gamma=0.996,
              lam=0.95,
              num_mini_batches=4,
              device=device,
              shuffle_batch=False,
              )

#load a model
checkpoint = torch.load(checkpoint_weight_path)
actor.architecture.load_state_dict(checkpoint['actor_architecture_state_dict'])
actor.distribution.load_state_dict(checkpoint['actor_distribution_state_dict'])
critic.architecture.load_state_dict(checkpoint['critic_architecture_state_dict'])
ppo.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

#save a model in script file
torch.jit.script(actor.architecture).save(weight_path + "actor_architecture_" + str(update_number) + ".pt")
# torch.jit.script(actor.distribution).save(weight_path + "actor_distribution_" + str(update_number) + ".pt")
# torch.jit.script(critic.architecture).save(weight_path + "critic_architecture_" + str(update_number) + ".pt")
# torch.jit.script(ppo.optimizer).save(weight_path + "ppo_optimizer_" + str(update_number) + ".pt")
