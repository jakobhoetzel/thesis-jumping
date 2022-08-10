from datetime import datetime
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from .storage import RolloutStorage
from adamp import AdamP


class PPO:
    def __init__(self,
                 actor_run,
                 actor_jump,
                 actor_manager,
                 critic_run,
                 critic_jump,
                 critic_manager,
                 estimator_run,
                 estimator_jump,
                 num_envs,
                 num_transitions_per_env,
                 num_learning_epochs,
                 num_mini_batches,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=5e-4,  # 0.5
                 estimator_loss_coef=0.1,
                 entropy_coef=0.0,
                 learning_rate=5e-4,
                 learning_rate_critic=5e-4,
                 sl_learning_rate=5e-6,
                 max_grad_norm=0.5,
                 use_clipped_value_loss=True,
                 log_dir='run',
                 device='cpu',
                 shuffle_batch=True,
                 manager_training=False):

        # PPO components
        self.actor_run = actor_run
        self.actor_jump = actor_jump
        self.actor_manager = actor_manager
        self.critic_run = critic_run
        self.critic_jump = critic_jump
        self.critic_manager = critic_manager
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_run.obs_shape, actor_jump.obs_shape,
                                      actor_manager.obs_shape,
                                      critic_run.obs_shape, critic_jump.obs_shape, critic_manager.obs_shape,
                                      actor_run.action_shape, device)

        if shuffle_batch:
            self.batch_sampler = self.storage.mini_batch_generator_shuffle
        else:
            self.batch_sampler = self.storage.mini_batch_generator_inorder

        self.optimizer = optim.Adam([*self.critic_run.parameters(), *self.critic_jump.parameters()], lr=learning_rate)
        self.device = device
        self.manager_training = False  # manager_training

        # env parameters
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        # self.estimator_loss_coef = estimator_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        # Log
        self.log_dir = os.path.join(log_dir, datetime.now().strftime('%b%d_%H-%M-%S'))
        self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=60)
        self.tot_timesteps = 0
        self.tot_time = 0

        # temps
        self.actions = None
        self.actions_log_prob = None
        self.actor_obs_run = None
        self.critic_obs_run = None
        self.actor_obs_jump = None
        self.actor_obs_manager = None
        self.run_bool = None
        self.jump_bool = None

    def observe(self, actor_obs_run, critic_obs_run, actor_obs_jump,
                run_bool_input=None):  # run_bool = 1 when running network active; run_bool = 0 when jumping network active
        self.actor_obs_run = actor_obs_run
        self.actor_obs_jump = actor_obs_jump
        self.critic_obs_run = critic_obs_run
        actions_log_prob_manager = None
        if run_bool_input is None:
            value_run = self.critic_run.predict(torch.from_numpy(self.critic_obs_run).to(self.device))
            value_jump = self.critic_jump.predict(torch.from_numpy(self.actor_obs_jump).to(self.device))
            self.run_bool = value_run > value_jump
        else:
            self.run_bool = torch.from_numpy(run_bool_input).to(self.device)
        self.jump_bool = torch.add(torch.ones(self.run_bool.size(), device=self.device), self.run_bool,
                                   alpha=-1)  # 1-run_bool

        actions_run, actions_run_log_prob = self.actor_run.sample(torch.from_numpy(actor_obs_run).to(self.device))
        actions_jump, actions_jump_log_prob = self.actor_jump.sample(torch.from_numpy(actor_obs_jump).to(self.device))
        self.actions = self.run_bool * actions_run + self.jump_bool * actions_jump

        if self.manager_training and actions_log_prob_manager is not None:
            self.actions_log_prob = actions_log_prob_manager
        else:
            self.actions_log_prob = self.run_bool[:, 0] * actions_run_log_prob + self.jump_bool[:, 0] * actions_jump_log_prob

        return self.actions.cpu().numpy(), self.run_bool.cpu().numpy()

    def step(self, value_obs_run, value_obs_jump, value_obs_manager, est_obs, robotState, rews, dones, guideline):
        value_run = self.critic_run.predict(torch.from_numpy(value_obs_run).to(self.device))
        value_jump = self.critic_jump.predict(torch.from_numpy(value_obs_jump).to(self.device))
        values = self.run_bool * value_run + self.jump_bool * value_jump
        self.storage.add_transitions(None, None, None, value_obs_run,
                                     value_obs_jump, value_run, value_jump, None, self.actions, None, None, rews, dones,
                                     values, self.actions_log_prob, self.run_bool, None)


    def update(self, actor_obs_manager, value_obs_run, value_obs_jump, value_obs_manager, log_this_iteration, update,
               actor_manager_update=True):
        value_run = self.critic_run.predict(torch.from_numpy(self.critic_obs_run).to(self.device))
        value_jump = self.critic_jump.predict(torch.from_numpy(self.actor_obs_jump).to(self.device))
        run_bool = value_run > value_jump
        jump_bool = torch.add(torch.ones(run_bool[:, 0:1].shape, device=self.device), run_bool[:, 0:1], alpha=-1)
        last_values = run_bool * value_run + jump_bool * value_jump

        # Learning step
        self.storage.compute_returns(last_values, run_bool, jump_bool, self.gamma, self.lam)
        mean_value_loss, mean_surrogate_loss, mean_entropy, infos = self._train_step(actor_manager_update)
        self.storage.clear()

        if log_this_iteration:
            self.log({**locals(), **infos, 'it': update})

    def log(self, variables, width=80, pad=28):
        self.tot_timesteps += self.num_transitions_per_env * self.num_envs

        self.writer.add_scalar('Loss/value_function', variables['mean_value_loss'], variables['it'])

    def _train_step(self, actor_manager_update=True):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        # mean_estimation_loss = 0
        mean_entropy = 0
        for epoch in range(self.num_learning_epochs):
            for critic_run_obs_batch, critic_jump_obs_batch, actions_batch, target_values_batch, \
                advantages_batch, returns_batch, old_actions_log_prob_batch, run_bool_batch,  guideline_batch\
                    in self.batch_sampler(self.num_mini_batches):

                jump_bool_batch = torch.add(torch.ones(run_bool_batch[:, 0:1].shape, device=self.device), run_bool_batch[:, 0:1], alpha=-1)
                value_batch = run_bool_batch * self.critic_run.evaluate(critic_run_obs_batch) \
                         + jump_bool_batch * self.critic_jump.evaluate(critic_jump_obs_batch)

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                loss = value_loss


            # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_([*self.critic_run.parameters(), *self.critic_jump.parameters()],
                                         self.max_grad_norm)
                self.optimizer.step()

                mean_value_loss += value_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates

        return mean_value_loss, mean_surrogate_loss, mean_entropy, locals()