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
                 critic_run,
                 critic_jump,
                 estimator,
                 num_envs,
                 num_transitions_per_env,
                 num_learning_epochs,
                 num_mini_batches,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=0.5,
                 estimator_loss_coef=0.1,
                 entropy_coef=0.0,
                 learning_rate=5e-4,
                 max_grad_norm=0.5,
                 use_clipped_value_loss=True,
                 log_dir='run',
                 device='cpu',
                 shuffle_batch=True):

        # PPO components
        self.actor_run = actor_run
        self.actor_jump = actor_jump
        self.critic_run = critic_run
        self.critic_jump = critic_jump
        self.estimator = estimator
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_run.obs_shape, critic_run.obs_shape, actor_run.action_shape, estimator.input_shape, estimator.output_shape, device)

        if shuffle_batch:
            self.batch_sampler = self.storage.mini_batch_generator_shuffle
        else:
            self.batch_sampler = self.storage.mini_batch_generator_inorder

        # self.optimizer = optim.Adam([*self.actor.parameters(), *self.critic.parameters()], lr=learning_rate)
        self.optimizer = AdamP([*self.actor_run.parameters(), *self.actor_jump.parameters(), *self.critic_run.parameters(), *self.critic_jump.parameters(), *self.estimator.parameters()], lr=learning_rate)
        self.device = device

        # env parameters
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.estimator_loss_coef = estimator_loss_coef
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
        self.actor_obs = None
        self.run_bool = None
        self.jump_bool = None

    def observe(self, actor_obs, run_bool):  # run_bool = 1 when running network active; run_bool = 0 when jumping network active
        self.actor_obs = actor_obs
        self.run_bool = torch.from_numpy(run_bool).to(self.device)
        self.jump_bool = torch.add(torch.ones(self.run_bool.size()).to(self.device), self.run_bool, alpha=-1)  # 1-run_bool

        actions_run, actions_run_log_prob = self.actor_run.sample(torch.from_numpy(actor_obs).to(self.device))
        actions_jump, actions_jump_log_prob = self.actor_jump.sample(torch.from_numpy(actor_obs).to(self.device))
        self.actions = self.run_bool * actions_run.to(self.device) + self.jump_bool * actions_jump.to(self.device)
        self.actions_log_prob = self.run_bool[:,0] * actions_run_log_prob.to(self.device) + self.jump_bool[:,0] * actions_jump_log_prob.to(self.device)
        # self.actions, self.actions_log_prob = run_bool * self.actor_run.sample(torch.from_numpy(actor_obs).to(self.device)) \
        #                                       + self.jump_bool * self.actor_jump.sample(torch.from_numpy(actor_obs).to(self.device))
        # self.actions = np.clip(self.actions.numpy(), self.env.action_space.low, self.env.action_space.high)
        return self.actions.cpu().numpy()

    def step(self, value_obs, est_in, robotState, rews, dones):
        values = self.run_bool[:,0:1] * self.critic_run.predict(torch.from_numpy(value_obs).to(self.device)) \
                 + self.jump_bool[:,0:1] * self.critic_jump.predict(torch.from_numpy(value_obs).to(self.device))
        self.storage.add_transitions(self.actor_obs, value_obs, self.actions, est_in, robotState, rews, dones, values,
                                     self.actions_log_prob, self.run_bool)

    def update(self, actor_obs, value_obs, log_this_iteration, update, run_bool):
        jump_bool = torch.add(torch.ones(run_bool[:,0:1].shape).to(self.device), torch.from_numpy(run_bool[:,0:1]).to(self.device), alpha=-1)
        # self.jump_bool = torch.add(torch.ones(self.run_bool.size()).to(self.device), self.run_bool, alpha=-1)  # 1-run_bool
        last_values = torch.from_numpy(run_bool[:,0:1]).to(self.device) * self.critic_run.predict(torch.from_numpy(value_obs).to(self.device)) \
                      + jump_bool * self.critic_jump.predict(torch.from_numpy(value_obs).to(self.device))

        # Learning step
        self.storage.compute_returns(last_values.to(self.device), self.gamma, self.lam)
        mean_value_loss, mean_surrogate_loss, mean_estimation_loss, mean_entropy, infos = self._train_step()
        self.storage.clear()

        if log_this_iteration:
            self.log({**locals(), **infos, 'it': update})

    def log(self, variables, width=80, pad=28):
        self.tot_timesteps += self.num_transitions_per_env * self.num_envs
        mean_std_run = self.actor_run.distribution.std.mean()
        mean_std_jump = self.actor_jump.distribution.std.mean()

        self.writer.add_scalar('Loss/value_function', variables['mean_value_loss'], variables['it'])
        self.writer.add_scalar('Loss/surrogate', variables['mean_surrogate_loss'], variables['it'])
        self.writer.add_scalar('Loss/estimation', variables['mean_estimation_loss'], variables['it'])
        self.writer.add_scalar('Policy/entropy', variables['mean_entropy'], variables['it'])
        self.writer.add_scalar('Policy/mean_noise_std_run', mean_std_run.item(), variables['it'])
        self.writer.add_scalar('Policy/mean_noise_std_jump', mean_std_jump.item(), variables['it'])

    def _train_step(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_estimation_loss = 0
        mean_entropy = 0
        for epoch in range(self.num_learning_epochs):
            for actor_obs_batch, critic_obs_batch, actions_batch, target_values_batch, est_in_batch, robotState_batch, \
                advantages_batch, returns_batch, old_actions_log_prob_batch, run_bool_batch \
                    in self.batch_sampler(self.num_mini_batches):

                jump_bool_batch = torch.add(torch.ones(run_bool_batch.size()).to(self.device), run_bool_batch, alpha=-1)

                actions_run_log_prob_batch, entropy_run_batch = self.actor_run.evaluate(actor_obs_batch, actions_batch)
                actions_jump_log_prob_batch, entropy_jump_batch = self.actor_jump.evaluate(actor_obs_batch, actions_batch)
                actions_log_prob_batch = run_bool_batch[:,0] * actions_run_log_prob_batch + jump_bool_batch[:,0] * actions_jump_log_prob_batch
                entropy_batch = run_bool_batch[:,0] * entropy_run_batch+ jump_bool_batch[:,0] * entropy_jump_batch
                # actions_log_prob_batch, entropy_batch = run_bool_batch * self.actor_run.evaluate(actor_obs_batch, actions_batch) \
                #                                         + jump_bool_batch * self.actor_jump.evaluate(actor_obs_batch, actions_batch)
                value_batch = run_bool_batch * self.critic_run.evaluate(critic_obs_batch) + (1-run_bool_batch) * self.critic_jump.evaluate(critic_obs_batch)
                estimation_batch = self.estimator.evaluate(est_in_batch)

                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                   1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                estimator_loss = (robotState_batch - estimation_batch).pow(2).mean()

                loss = surrogate_loss + self.value_loss_coef * value_loss + self.estimator_loss_coef * estimator_loss\
                       - self.entropy_coef * entropy_batch.mean()

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_([*self.actor_run.parameters(), *self.actor_jump.parameters(), *self.critic_run.parameters(), *self.critic_jump.parameters()], self.max_grad_norm)
                self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                mean_estimation_loss += estimator_loss.item()
                mean_entropy += entropy_batch.mean()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_estimation_loss /= num_updates
        mean_entropy /= num_updates

        return mean_value_loss, mean_surrogate_loss, mean_estimation_loss, mean_entropy, locals()
