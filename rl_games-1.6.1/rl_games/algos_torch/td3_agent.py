import copy
import os
import time

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from rl_games.algos_torch import model_builder, torch_ext
from rl_games.common import experience, vecenv
from rl_games.common.a2c_common import print_statistics
from rl_games.interfaces.base_algorithm import BaseAlgorithm


class TD3Agent(BaseAlgorithm):
    """
    Twin Delayed Deep Deterministic Policy Gradient (TD3).

    Differences from SAC:
    - Deterministic actor (use dist.mean, no sampling)
    - No entropy / alpha
    - Exploration via additive Gaussian noise
    - Target policy smoothing: clipped noise on target actor during critic update
    - Delayed actor + target updates every `policy_delay` critic steps
    - Separate actor_target network (soft-tracked copy of actor)
    """

    def __init__(self, base_name, params):
        self.config = config = params['config']

        self.load_networks(params)
        self.base_init(base_name, config)

        # ── Shared hyper-params ──────────────────────────────────────────────
        self.num_warmup_steps   = config["num_warmup_steps"]
        self.gamma              = config["gamma"]
        self.critic_tau         = float(config["critic_tau"])
        self.batch_size         = config["batch_size"]
        self.replay_buffer_size = config["replay_buffer_size"]
        self.num_steps_per_episode = config.get("num_steps_per_episode", 1)
        self.normalize_input    = config.get("normalize_input", False)
        self.max_env_steps      = config.get("max_env_steps", 1000)

        # ── TD3-specific hyper-params ────────────────────────────────────────
        self.policy_delay  = config.get("policy_delay", 2)      # actor update interval
        self.expl_noise    = config.get("expl_noise", 0.1)      # exploration noise std
        self.policy_noise  = config.get("policy_noise", 0.2)    # target smoothing noise std
        self.noise_clip    = config.get("noise_clip", 0.5)      # target smoothing noise clip
        self._critic_steps = 0                                   # counts critic updates

        self.num_frames_per_epoch = self.num_actors * self.num_steps_per_episode

        action_space       = self.env_info['action_space']
        self.actions_num   = action_space.shape[0]
        self.action_range  = [
            float(self.env_info['action_space'].low.min()),
            float(self.env_info['action_space'].high.max()),
        ]

        obs_shape  = torch_ext.shape_whc_to_cwh(self.obs_shape)
        net_config = {
            'obs_dim':         self.env_info["observation_space"].shape[0],
            'action_dim':      self.env_info["action_space"].shape[0],
            'actions_num':     self.actions_num,
            'input_shape':     obs_shape,
            'normalize_input': self.normalize_input,
        }
        self.model = self.network.build(net_config)
        self.model.to(self._device)

        # ── Actor target: frozen slow-moving copy ────────────────────────────
        self.actor_target = copy.deepcopy(self.model.sac_network.actor)
        self.actor_target.to(self._device)
        for p in self.actor_target.parameters():
            p.requires_grad = False

        print(f"[TD3] actors={self.num_actors}  batch={self.batch_size}  "
              f"policy_delay={self.policy_delay}  expl_noise={self.expl_noise}")

        # ── Optimisers ───────────────────────────────────────────────────────
        self.actor_optimizer = torch.optim.Adam(
            self.model.sac_network.actor.parameters(),
            lr=float(config['actor_lr']),
            betas=config.get("actor_betas", [0.9, 0.999]),
        )
        self.critic_optimizer = torch.optim.Adam(
            self.model.sac_network.critic.parameters(),
            lr=float(config["critic_lr"]),
            betas=config.get("critic_betas", [0.9, 0.999]),
        )

        self.replay_buffer = experience.VectorizedReplayBuffer(
            self.env_info['observation_space'].shape,
            self.env_info['action_space'].shape,
            self.replay_buffer_size,
            self._device,
        )

        self.c_loss        = nn.MSELoss()
        self.algo_observer = config['features']['observer']

    # ── Setup ─────────────────────────────────────────────────────────────────

    def load_networks(self, params):
        builder = model_builder.ModelBuilder()
        self.config['network'] = builder.load(params)

    def base_init(self, base_name, config):
        self.env_config = config.get('env_config', {})
        self.num_actors = config.get('num_actors', 1)
        self.env_name   = config['env_name']

        self.env_info = config.get('env_info')
        if self.env_info is None:
            self.vec_env  = vecenv.create_vec_env(self.env_name, self.num_actors, **self.env_config)
            self.env_info = self.vec_env.get_env_info()

        self._device    = config.get('device', 'cuda:0')
        self.ppo_device = self._device

        self.rewards_shaper    = config['reward_shaper']
        self.observation_space = self.env_info['observation_space']
        self.weight_decay      = config.get('weight_decay', 0.0)
        self.is_train          = config.get('is_train', True)

        self.save_best_after = config.get('save_best_after', 500)
        self.print_stats     = config.get('print_stats', True)
        self.rnn_states      = None
        self.name            = base_name
        self.max_epochs      = config.get('max_epochs', -1)
        self.max_frames      = config.get('max_frames', -1)
        self.save_freq       = config.get('save_frequency', 0)

        self.network        = config['network']
        self.rewards_shaper = config['reward_shaper']
        self.num_agents     = self.env_info.get('agents', 1)
        self.obs_shape      = self.observation_space.shape

        self.games_to_track = config.get('games_to_track', 100)
        self.game_rewards   = torch_ext.AverageMeter(1, self.games_to_track).to(self._device)
        self.game_lengths   = torch_ext.AverageMeter(1, self.games_to_track).to(self._device)
        self.obs            = None

        self.frame        = 0
        self.epoch_num    = 0
        self.update_time  = 0
        self.last_mean_rewards = -1_000_000_000
        self.play_time    = 0

        pbt_str = ''
        self.population_based_training = config.get('population_based_training', False)
        if self.population_based_training:
            pbt_str = f'_pbt_{config["pbt_idx"]:02d}'
        full_experiment_name = config.get('full_experiment_name', None)
        if full_experiment_name:
            self.experiment_name = full_experiment_name
        else:
            self.experiment_name = config['name'] + pbt_str + datetime.now().strftime("_%d-%H-%M-%S")

        self.train_dir      = config.get('train_dir', 'runs')
        self.experiment_dir = os.path.join(self.train_dir, self.experiment_name)
        self.nn_dir         = os.path.join(self.experiment_dir, 'nn')
        self.summaries_dir  = os.path.join(self.experiment_dir, 'summaries')
        os.makedirs(self.nn_dir,       exist_ok=True)
        os.makedirs(self.summaries_dir, exist_ok=True)

        self.writer = SummaryWriter('runs/' + config['name'] + datetime.now().strftime("_%d-%H-%M-%S"))
        print("Run Directory:", config['name'] + datetime.now().strftime("_%d-%H-%M-%S"))

        self.is_tensor_obses   = False
        self.is_rnn            = False
        self.last_rnn_indices  = None
        self.last_state_indices = None

    def init_tensors(self):
        batch_size = self.num_agents * self.num_actors
        self.current_rewards = torch.zeros(batch_size, dtype=torch.float32, device=self._device)
        self.current_lengths = torch.zeros(batch_size, dtype=torch.long,    device=self._device)
        self.dones           = torch.zeros(batch_size, dtype=torch.uint8,   device=self._device)

    # ── Observation helpers (identical to SAC) ────────────────────────────────

    def preproc_obs(self, obs):
        if isinstance(obs, dict):
            obs = obs['obs']
        return self.model.norm_obs(obs)

    def cast_obs(self, obs):
        if isinstance(obs, torch.Tensor):
            self.is_tensor_obses = True
        elif isinstance(obs, np.ndarray):
            if self.observation_space.dtype == np.uint8:
                obs = torch.ByteTensor(obs).to(self._device)
            else:
                obs = torch.FloatTensor(obs).to(self._device)
        return obs

    def obs_to_tensors(self, obs):
        obs_is_dict = isinstance(obs, dict)
        if obs_is_dict:
            upd_obs = {k: self._obs_to_tensors_internal(v) for k, v in obs.items()}
        else:
            upd_obs = self.cast_obs(obs)
        if not obs_is_dict or 'obs' not in obs:
            upd_obs = {'obs': upd_obs}
        return upd_obs

    def _obs_to_tensors_internal(self, obs):
        if isinstance(obs, dict):
            return {k: self._obs_to_tensors_internal(v) for k, v in obs.items()}
        return self.cast_obs(obs)

    def preprocess_actions(self, actions):
        if not self.is_tensor_obses:
            actions = actions.cpu().numpy()
        return actions

    # ── Env helpers ───────────────────────────────────────────────────────────

    def env_step(self, actions):
        actions = self.preprocess_actions(actions)
        obs, rewards, dones, infos = self.vec_env.step(actions)
        if self.is_tensor_obses:
            return self.obs_to_tensors(obs), rewards.to(self._device), dones.to(self._device), infos
        return (torch.from_numpy(obs).to(self._device),
                torch.from_numpy(rewards).to(self._device),
                torch.from_numpy(dones).to(self._device), infos)

    def env_reset(self):
        with torch.no_grad():
            obs = self.vec_env.reset()
        return self.obs_to_tensors(obs)

    # ── Policy helpers ────────────────────────────────────────────────────────

    def act_deterministic(self, obs):
        """Deterministic action from current actor (no noise, no sampling)."""
        obs  = self.preproc_obs(obs)
        dist = self.model.sac_network.actor(obs)
        return dist.mean.clamp(*self.action_range)

    def act_target_det(self, obs_preprocessed):
        """Deterministic action from target actor (obs already preprocessed)."""
        dist = self.actor_target(obs_preprocessed)
        return dist.mean

    # ── Update logic ──────────────────────────────────────────────────────────

    def update_critic(self, obs, action, reward, next_obs, not_done):
        with torch.no_grad():
            # Target policy smoothing
            next_action = self.act_target_det(next_obs)
            noise = (torch.randn_like(next_action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip)
            next_action = (next_action + noise).clamp(*self.action_range)

            target_Q1, target_Q2 = self.model.critic_target(next_obs, next_action)
            target_Q = reward + not_done * self.gamma * torch.min(target_Q1, target_Q2)

        current_Q1, current_Q2 = self.model.critic(obs, action)
        critic_loss = self.c_loss(current_Q1, target_Q) + self.c_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        return critic_loss.detach(), \
               self.c_loss(current_Q1, target_Q).detach(), \
               self.c_loss(current_Q2, target_Q).detach()

    def update_actor(self, obs):
        # Freeze critic during actor update
        for p in self.model.sac_network.critic.parameters():
            p.requires_grad = False

        action     = self.act_deterministic(obs)
        Q1, Q2     = self.model.critic(obs, action)
        actor_loss = -torch.min(Q1, Q2).mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        for p in self.model.sac_network.critic.parameters():
            p.requires_grad = True

        return actor_loss.detach()

    def soft_update(self, src, tgt, tau):
        for sp, tp in zip(src.parameters(), tgt.parameters()):
            tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)

    def update(self, step):
        obs, action, reward, next_obs, done = self.replay_buffer.sample(self.batch_size)
        not_done = (~done).float()

        obs      = self.preproc_obs(obs)
        next_obs = self.preproc_obs(next_obs)

        critic_loss, c1_loss, c2_loss = self.update_critic(obs, action, reward, next_obs, not_done)
        self._critic_steps += 1

        actor_loss = None
        if self._critic_steps % self.policy_delay == 0:
            actor_loss = self.update_actor(obs)
            # Soft-update both targets
            self.soft_update(self.model.sac_network.actor,   self.actor_target,                         self.critic_tau)
            self.soft_update(self.model.sac_network.critic,  self.model.sac_network.critic_target,      self.critic_tau)

        return actor_loss, c1_loss, c2_loss

    # ── Training loop ─────────────────────────────────────────────────────────

    @property
    def device(self):
        return self._device

    def set_eval(self):  self.model.eval()
    def set_train(self): self.model.train()

    def play_steps(self, random_exploration=False):
        total_time_start  = time.time()
        total_update_time = 0
        step_time         = 0.0
        actor_losses, critic1_losses, critic2_losses = [], [], []

        obs = self.obs
        if isinstance(obs, dict):
            obs = obs['obs']

        for _ in range(self.num_steps_per_episode):
            self.set_eval()

            if random_exploration:
                action = torch.rand(
                    (self.num_actors, *self.env_info["action_space"].shape),
                    device=self._device) * 2.0 - 1.0
            else:
                with torch.no_grad():
                    action = self.act_deterministic(obs)
                    noise  = torch.randn_like(action) * self.expl_noise
                    action = (action + noise).clamp(*self.action_range)

            t0 = time.time()
            with torch.no_grad():
                next_obs, rewards, dones, infos = self.env_step(action)
            step_time += time.time() - t0

            self.current_rewards += rewards
            self.current_lengths += 1

            all_done_indices = dones.nonzero(as_tuple=False)
            done_indices     = all_done_indices[::self.num_agents]
            self.game_rewards.update(self.current_rewards[done_indices])
            self.game_lengths.update(self.current_lengths[done_indices])

            not_dones = 1.0 - dones.float()
            self.algo_observer.process_infos(infos, done_indices)

            no_timeouts = self.current_lengths != self.max_env_steps
            dones_masked = dones * no_timeouts

            self.current_rewards = self.current_rewards * not_dones
            self.current_lengths = self.current_lengths * not_dones

            if isinstance(next_obs, dict):
                next_obs_t = next_obs['obs']
            else:
                next_obs_t = next_obs

            self.obs = next_obs

            shaped_rewards = self.rewards_shaper(rewards)
            self.replay_buffer.add(
                obs, action,
                torch.unsqueeze(shaped_rewards, 1),
                next_obs_t,
                torch.unsqueeze(dones_masked, 1),
            )
            obs = next_obs_t

            if not random_exploration:
                self.set_train()
                t1 = time.time()
                actor_loss, c1_loss, c2_loss = self.update(self.epoch_num)
                total_update_time += time.time() - t1

                if actor_loss is not None:
                    actor_losses.append(actor_loss)
                critic1_losses.append(c1_loss)
                critic2_losses.append(c2_loss)

        total_time = time.time() - total_time_start
        play_time  = total_time - total_update_time
        return step_time, play_time, total_update_time, total_time, actor_losses, critic1_losses, critic2_losses

    def train_epoch(self):
        random_exploration = self.epoch_num < self.num_warmup_steps
        return self.play_steps(random_exploration)

    def train(self):
        self.init_tensors()
        self.algo_observer.after_init(self)
        total_time = 0
        self.obs    = self.env_reset()

        while True:
            self.epoch_num += 1
            step_time, play_time, update_time, epoch_time, \
                actor_losses, critic1_losses, critic2_losses = self.train_epoch()

            total_time  += epoch_time
            curr_frames  = self.num_frames_per_epoch
            self.frame  += curr_frames

            fps_step  = curr_frames / max(step_time,  1e-6)
            fps_total = curr_frames / max(epoch_time, 1e-6)

            print_statistics(self.print_stats, curr_frames, step_time, play_time, epoch_time,
                             self.epoch_num, self.max_epochs, self.frame, self.max_frames)

            self.writer.add_scalar('performance/step_fps',    fps_step,  self.frame)
            self.writer.add_scalar('performance/total_fps',   fps_total, self.frame)
            self.writer.add_scalar('performance/update_time', update_time, self.frame)

            if self.epoch_num >= self.num_warmup_steps:
                self.writer.add_scalar('losses/c1_loss', torch_ext.mean_list(critic1_losses).item(), self.frame)
                self.writer.add_scalar('losses/c2_loss', torch_ext.mean_list(critic2_losses).item(), self.frame)
                if actor_losses:
                    self.writer.add_scalar('losses/a_loss', torch_ext.mean_list(actor_losses).item(), self.frame)

            self.writer.add_scalar('info/epochs', self.epoch_num, self.frame)
            self.algo_observer.after_print_stats(self.frame, self.epoch_num, total_time)

            if self.game_rewards.current_size > 0:
                mean_rewards = self.game_rewards.get_mean()
                mean_lengths = self.game_lengths.get_mean()

                self.writer.add_scalar('rewards/step',           mean_rewards, self.frame)
                self.writer.add_scalar('episode_lengths/step',   mean_lengths, self.frame)

                checkpoint_name = self.config['name'] + '_ep_' + str(self.epoch_num) + '_rew_' + str(mean_rewards)
                should_exit = False

                if self.save_freq > 0 and self.epoch_num % self.save_freq == 0:
                    self.save(os.path.join(self.nn_dir, 'last_' + checkpoint_name))

                if mean_rewards > self.last_mean_rewards and self.epoch_num >= self.save_best_after:
                    print('saving best rewards:', mean_rewards)
                    self.last_mean_rewards = mean_rewards
                    self.save(os.path.join(self.nn_dir, self.config['name']))
                    if self.last_mean_rewards > self.config.get('score_to_win', float('inf')):
                        self.save(os.path.join(self.nn_dir, checkpoint_name))
                        should_exit = True

                if self.epoch_num >= self.max_epochs != -1:
                    self.save(os.path.join(self.nn_dir,
                        'last_' + self.config['name'] + '_ep_' + str(self.epoch_num)
                        + '_rew_' + str(mean_rewards).replace('[', '_').replace(']', '_')))
                    print('MAX EPOCHS NUM!')
                    should_exit = True

                if should_exit:
                    return self.last_mean_rewards, self.epoch_num

    # ── Checkpoint ────────────────────────────────────────────────────────────

    def get_weights(self):
        return {
            'actor':         self.model.sac_network.actor.state_dict(),
            'actor_target':  self.actor_target.state_dict(),
            'critic':        self.model.sac_network.critic.state_dict(),
            'critic_target': self.model.sac_network.critic_target.state_dict(),
        }

    def get_full_state_weights(self):
        state = self.get_weights()
        state['epoch']            = self.epoch_num
        state['frame']            = self.frame
        state['actor_optimizer']  = self.actor_optimizer.state_dict()
        state['critic_optimizer'] = self.critic_optimizer.state_dict()
        if self.normalize_input:
            state['running_mean_std'] = self.model.running_mean_std.state_dict()
        return state

    def set_weights(self, weights):
        self.model.sac_network.actor.load_state_dict(weights['actor'])
        self.actor_target.load_state_dict(weights['actor_target'])
        self.model.sac_network.critic.load_state_dict(weights['critic'])
        self.model.sac_network.critic_target.load_state_dict(weights['critic_target'])
        if self.normalize_input and 'running_mean_std' in weights:
            self.model.running_mean_std.load_state_dict(weights['running_mean_std'])

    def set_full_state_weights(self, weights, set_epoch=True):
        self.set_weights(weights)
        if set_epoch:
            self.epoch_num = weights['epoch']
            self.frame     = weights['frame']
        self.actor_optimizer.load_state_dict(weights['actor_optimizer'])
        self.critic_optimizer.load_state_dict(weights['critic_optimizer'])
        self.last_mean_rewards = weights.get('last_mean_rewards', -1_000_000_000)

    def save(self, fn):
        torch_ext.save_checkpoint(fn, self.get_full_state_weights())

    def restore(self, fn, set_epoch=True):
        checkpoint = torch_ext.load_checkpoint(fn)
        self.set_full_state_weights(checkpoint, set_epoch=set_epoch)

    def get_param(self, param_name):     pass
    def set_param(self, param_name, v):  pass
    def get_masked_action_values(self, obs, masks): assert False
    def clear_stats(self):
        self.game_rewards.clear()
        self.game_lengths.clear()
        self.mean_rewards = self.last_mean_rewards = -1_000_000_000
        self.algo_observer.after_clear_stats()
