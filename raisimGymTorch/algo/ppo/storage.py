import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class RolloutStorage:
    # def __init__(self, num_envs, num_transitions_per_env, actor_run_obs_shape, actor_jump_obs_shape, actor_manager_obs_shape,
    #              critic_obs_run_shape, critic_obs_jump_shape, critic_manager_obs_shape, actions_shape, estimator_input_shape, robotState_shape, device):
    def __init__(self, num_envs, num_transitions_per_env, actor_run_obs_shape, actor_jump_obs_shape, actor_manager_obs_shape,
                 critic_obs_run_shape, critic_obs_jump_shape, critic_manager_obs_shape, actions_shape, device):
        self.device = device

        # Core
        self.critic_run_obs = torch.zeros(num_transitions_per_env, num_envs, *critic_obs_run_shape, device=self.device)
        self.critic_jump_obs = torch.zeros(num_transitions_per_env, num_envs, *critic_obs_jump_shape, device=self.device)
        # self.critic_manager_obs = torch.zeros(num_transitions_per_env, num_envs, *critic_manager_obs_shape, device=self.device)
        # self.actor_run_obs = torch.zeros(num_transitions_per_env, num_envs, *actor_run_obs_shape, device=self.device)
        # self.actor_jump_obs = torch.zeros(num_transitions_per_env, num_envs, *actor_jump_obs_shape, device=self.device)
        # self.actor_manager_obs = torch.zeros(num_transitions_per_env, num_envs, *actor_manager_obs_shape, device=self.device)
        # self.estimator_input = torch.zeros(num_transitions_per_env, num_envs, *estimator_input_shape, device=self.device)
        # self.robotState = torch.zeros(num_transitions_per_env, num_envs, *robotState_shape, device=self.device)
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()
        self.run_bool = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.guideline = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)

        # For PPO
        self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        self.step = 0

    def add_transitions(self, actor_run_obs, actor_jump_obs, actor_manager_obs, critic_run_obs, critic_jump_obs, critic_manager_obs,
                        actions, est_in, robotState, rewards, dones, values, actions_log_prob, run_bool, guideline):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        self.critic_run_obs[self.step].copy_(torch.from_numpy(critic_run_obs).to(self.device))
        self.critic_jump_obs[self.step].copy_(torch.from_numpy(critic_jump_obs).to(self.device))
        # self.critic_manager_obs[self.step].copy_(torch.from_numpy(critic_manager_obs).to(self.device))
        # self.actor_run_obs[self.step].copy_(torch.from_numpy(actor_run_obs).to(self.device))
        # self.actor_jump_obs[self.step].copy_(torch.from_numpy(actor_jump_obs).to(self.device))
        # self.actor_manager_obs[self.step].copy_(torch.from_numpy(actor_manager_obs).to(self.device))
        # self.estimator_input[self.step].copy_(torch.from_numpy(est_in).to(self.device))
        # self.robotState[self.step].copy_(torch.from_numpy(robotState).to(self.device))
        self.actions[self.step].copy_(actions.to(self.device))
        self.run_bool[self.step].copy_(run_bool.to(self.device))
        self.guideline[self.step].copy_(run_bool.to(self.device))
        self.rewards[self.step].copy_(torch.from_numpy(rewards).view(-1, 1).to(self.device))
        self.dones[self.step].copy_(torch.from_numpy(dones).view(-1, 1).to(self.device))
        self.values[self.step].copy_(values.to(self.device))
        self.actions_log_prob[self.step].copy_(actions_log_prob.view(-1, 1).to(self.device))
        self.step += 1

    def clear(self):
        self.step = 0

    def compute_returns(self, last_values, gamma, lam):
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):  # TODO: mix values (e.g. 1 step TD)
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
                # next_is_not_terminal = 1.0 - self.dones[step].float()
            else:
                next_values = self.values[step + 1]
                # next_is_not_terminal = 1.0 - self.dones[step+1].float()

            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]

        # Compute and normalize the advantages
        self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def mini_batch_generator_shuffle(self, num_mini_batches):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches

        for indices in BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=True):
            # actor_run_obs_batch = self.actor_run_obs.view(-1, *self.actor_run_obs.size()[2:])[indices]
            # actor_jump_obs_batch = self.actor_jump_obs.view(-1, *self.actor_jump_obs.size()[2:])[indices]
            # actor_manager_obs_batch = self.actor_manager_obs.view(-1, *self.actor_manager_obs.size()[2:])[indices]
            critic_run_obs_batch = self.critic_run_obs.view(-1, *self.critic_run_obs.size()[2:])[indices]
            critic_jump_obs_batch = self.critic_jump_obs.view(-1, *self.critic_jump_obs.size()[2:])[indices]
            # critic_manager_obs_batch = self.critic_manager_obs.view(-1, *self.critic_manager_obs.size()[2:])[indices]
            # est_in_batch = self.estimator_input.view(-1, *self.estimator_input.size()[2:])[indices]
            # robotState_batch = self.robotState.view(-1, *self.robotState.size()[2:])[indices]
            actions_batch = self.actions.view(-1, self.actions.size(-1))[indices]
            values_batch = self.values.view(-1, 1)[indices]
            returns_batch = self.returns.view(-1, 1)[indices]
            old_actions_log_prob_batch = self.actions_log_prob.view(-1, 1)[indices]
            advantages_batch = self.advantages.view(-1, 1)[indices]
            run_bool_batch = self.run_bool.view(-1, 1)[indices]
            guideline_batch = self.guideline.view(-1, 1)[indices]
        yield critic_run_obs_batch, critic_jump_obs_batch, actions_batch, values_batch, \
              advantages_batch, returns_batch, old_actions_log_prob_batch, run_bool_batch, guideline_batch

    def mini_batch_generator_inorder(self, num_mini_batches):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches

        for batch_id in range(num_mini_batches):
            yield self.critic_run_obs.view(-1, *self.critic_run_obs.size()[2:])[batch_id*mini_batch_size:(batch_id+1)*mini_batch_size], \
                self.critic_jump_obs.view(-1, *self.critic_jump_obs.size()[2:])[batch_id*mini_batch_size:(batch_id+1)*mini_batch_size], \
                self.actions.view(-1, self.actions.size(-1))[batch_id*mini_batch_size:(batch_id+1)*mini_batch_size], \
                self.values.view(-1, 1)[batch_id*mini_batch_size:(batch_id+1)*mini_batch_size], \
                self.advantages.view(-1, 1)[batch_id*mini_batch_size:(batch_id+1)*mini_batch_size], \
                self.returns.view(-1, 1)[batch_id*mini_batch_size:(batch_id+1)*mini_batch_size], \
                self.actions_log_prob.view(-1, 1)[batch_id*mini_batch_size:(batch_id+1)*mini_batch_size], \
                self.run_bool.view(-1, 1)[batch_id*mini_batch_size:(batch_id+1)*mini_batch_size], \
                self.guideline.view(-1, 1)[batch_id*mini_batch_size:(batch_id+1)*mini_batch_size]