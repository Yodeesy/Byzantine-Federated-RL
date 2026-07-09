import torch
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete
from policy import MlpPolicy, DiagonalGaussianMlpPolicy
from utils import get_inner_model, save_frames_as_gif
from utils import env_wrapper
import random

def _reset_env(env, seed=None):
    """Reset environment, compatible with both Gym (returns obs) and
    Gymnasium (returns (obs, info))."""
    if seed is not None:
        try:
            result = env.reset(seed=seed)
        except TypeError:
            # Old Gym: reset() doesn't accept seed kwarg
            env.seed(seed)
            result = env.reset()
    else:
        result = env.reset()
    if isinstance(result, tuple):
        return result[0]  # Gymnasium: (obs, info)
    return result           # Old Gym: obs


def _step_env(env, action):
    """Step environment, compatible with both Gym (returns 4-tuple) and
    Gymnasium (returns 5-tuple). Always returns (obs, rew, done, info)."""
    result = env.step(action)
    if len(result) == 5:
        # Gymnasium: (obs, rew, terminated, truncated, info)
        obs, rew, terminated, truncated, info = result
        done = terminated or truncated
        return obs, rew, done, info
    else:
        # Old Gym: (obs, rew, done, info)
        return result


class Worker:

    def __init__(self,
                 id,
                 is_Byzantine,
                 env_name,
                 hidden_units,
                 gamma,
                 activation = 'Tanh',
                 output_activation = 'Identity',
                 attack_type = None,
                 max_epi_len = 0,
                 opts = None
                 ):
        super(Worker, self).__init__()
        
        # setup
        self.id = id
        self.is_Byzantine = is_Byzantine
        self.gamma = gamma
        # make environment, check spaces, get obs / act dims
        self.env_name = env_name
        from utils import resolve_env_name
        self.env = gym.make(resolve_env_name(env_name))
        self.attack_type = attack_type
        self.max_epi_len = max_epi_len
        
        assert opts is not None
        self.opts = opts  # store for divergence-attack access

        # get observation dim
        obs_dim = self.env.observation_space.shape[0]
        if isinstance(self.env.action_space, Discrete):
            n_acts = self.env.action_space.n
        else:
            n_acts = self.env.action_space.shape[0]
        
        hidden_sizes = list(eval(hidden_units))
        self.sizes = [obs_dim]+hidden_sizes+[n_acts] # make core of policy network
        
        # get policy net
        if isinstance(self.env.action_space, Discrete):
            self.logits_net = MlpPolicy(self.sizes, activation, output_activation)
        else:
            self.logits_net = DiagonalGaussianMlpPolicy(self.sizes, activation,)
        
        if self.id == 1:
            print(self.logits_net)

    
    def load_param_from_master(self, param):
        model_actor = get_inner_model(self.logits_net)
        model_actor.load_state_dict({**model_actor.state_dict(), **param})

    def rollout(self, device, max_steps = 1000, render = False, env = None, obs = None, sample = True, mode = 'human', save_dir = './', filename = '.'):
        
        if env is None and obs is None:
            env = self.env
            obs = _reset_env(env)
            
        done = False  
        ep_rew = []
        frames = []
        step = 0
        while not done and step < max_steps:
            step += 1
            if render:
                if mode == 'rgb':
                    frames.append(env.render(mode="rgb_array"))
                else:
                    env.render()
                
            obs = env_wrapper(env.unwrapped.spec.id, obs)
            action = self.logits_net(torch.as_tensor(obs, dtype=torch.float32).to(device), sample = sample)[0]
            obs, rew, done, _ = _step_env(env, action)
            ep_rew.append(rew)

        if mode == 'rgb': save_frames_as_gif(frames, save_dir, filename)
        return np.sum(ep_rew), len(ep_rew), ep_rew
    
    def collect_experience_for_training(self, B, device, record = False, sample = True, attack_type = None):
        # make some empty lists for logging.
        batch_weights = []      # for R(tau) weighting in policy gradient
        batch_rets = []         # for measuring episode returns
        batch_lens = []         # for measuring episode lengths
        batch_log_prob = []     # for gradient computing

        # reset episode-specific variables
        obs = _reset_env(self.env)  # first obs comes from starting distribution
        done = False            # signal from environment that episode is over
        ep_rews = []            # list for rewards accrued throughout ep
        
        # make two lists for recording the trajectory
        if record:
            batch_states = []
            batch_actions = []

        t = 1
        # collect experience by acting in the environment with current policy
        while True:
            # save trajectory
            if record:
                batch_states.append(obs)
            # act in the environment  
            obs = env_wrapper(self.env_name, obs)
            
            # simulate random-action attacker if needed
            if self.is_Byzantine and attack_type is not None and self.attack_type == 'random-action':
                act_rnd = self.env.action_space.sample()
                if isinstance(act_rnd, int): # discrete action space
                    act_rnd = 0
                else: # continuous
                    act_rnd = np.zeros(len(self.env.action_space.sample()), dtype=np.float32) 
                act, log_prob = self.logits_net(torch.as_tensor(obs, dtype=torch.float32).to(device), sample = sample, fixed_action = act_rnd)
            else:
                act, log_prob = self.logits_net(torch.as_tensor(obs, dtype=torch.float32).to(device), sample = sample)
           
            obs, rew, done, info = _step_env(self.env, act)
            
            # simulate reward-flipping attacker if needed
            if self.is_Byzantine and attack_type is not None and self.attack_type == 'reward-flipping': 
                rew = - rew
                
            # timestep
            t = t + 1
            
            # save action_log_prob, reward
            batch_log_prob.append(log_prob)
            
            ep_rews.append(rew)
            
            # save trajectory
            if record:
                batch_actions.append(act)

            if done or len(ep_rews) >= self.max_epi_len:
                
                # if episode is over, record info about episode
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)
                
                # the weight for each logprob(a_t|s_T) is sum_t^T (gamma^(t'-t) * r_t')
                returns = []
                R = 0
                # simulate random-reware attacker if needed
                if self.is_Byzantine and attack_type is not None and self.attack_type == 'random-reward': 
                    random.shuffle(ep_rews)
                    for r in ep_rews:
                        R = r + self.gamma * R
                        returns.insert(0, R)
                else:
                    for r in ep_rews[::-1]:
                        R = r + self.gamma * R
                        returns.insert(0, R)            
                returns = torch.tensor(returns, dtype=torch.float32)
                
                # return whitening
                advantage = (returns - returns.mean()) / (returns.std() + 1e-20)
                batch_weights += advantage

                # end experience loop if we have enough of it
                if len(batch_lens) >= B:
                    break
                
                # reset episode-specific variables
                obs, done, ep_rews, t = _reset_env(self.env), False, [], 1


        # make torch tensor and restrict to batch_size
        weights = torch.as_tensor(batch_weights, dtype = torch.float32).to(device)
        logp = torch.stack(batch_log_prob)

        if record:
            return weights, logp, batch_rets, batch_lens, batch_states, batch_actions
        else:
            return weights, logp, batch_rets, batch_lens
    
    
    def train_one_epoch(self, B, device, sample):
        opts = getattr(self, 'opts', None)

        # ---- Divergence Attack: collect states for local high-entropy poisoning ----
        if self.is_Byzantine and self.attack_type == 'divergence-attack':
            (weights, logp, batch_rets, batch_lens,
             batch_states, _) = self.collect_experience_for_training(
                B, device, sample=sample, record=True)
        else:
            weights, logp, batch_rets, batch_lens = self.collect_experience_for_training(
                B, device, sample=sample, attack_type=self.attack_type)

        # calculate policy gradient loss
        batch_loss = -(logp * weights).mean()

        # take a single policy gradient update step
        self.logits_net.zero_grad()
        batch_loss.backward()

        # ---- Decentralized Boundary-Split Attack (DBSA): Two-Stage Local Optimization ----
        if self.is_Byzantine and self.attack_type == 'divergence-attack':
            from torch.distributions.categorical import Categorical
            import math

            # Stage I: 计算 \Delta_local(s) 并寻找脆弱状态
            delta_locals = []
            for obs in batch_states:
                obs_t = torch.as_tensor(obs, dtype=torch.float32).to(device)
                logits = self.logits_net.logits_net(obs_t)
                probs = torch.softmax(logits, dim=-1)

                top2_probs, top2_indices = torch.topk(probs, 2, dim=-1)
                delta = top2_probs[0] - top2_probs[1]
                delta_locals.append((delta.item(), obs_t, top2_indices[1].item()))

            delta_locals.sort(key=lambda x: x[0])
            num_triggers = max(1, int(len(delta_locals) * 0.1))
            trigger_data = delta_locals[:num_triggers]

            if len(trigger_data) > 0:
                honest_grad = [p.grad.clone() for p in self.parameters()]

                self.logits_net.zero_grad()
                loss_syn = 0
                for _, obs_t, top2_action in trigger_data:
                    logits_t = self.logits_net.logits_net(obs_t)
                    dist_t = Categorical(logits=logits_t)
                    logp_target = dist_t.log_prob(torch.tensor(top2_action, device=device))
                    loss_syn -= logp_target  # 最大化局部次优动作概率

                loss_syn = loss_syn / len(trigger_data)
                loss_syn.backward()

                # g_syn 就是反传得到的 malicious_grad
                malicious_grad = [p.grad.clone() for p in self.parameters()]

                # ==========================================================
                # Stage II: 解析解优化 (Closed-form Optimization)
                # ==========================================================

                # 使用组内大小 world_k，而不是全局 num_worker
                world_k = opts.num_worker // opts.num_groups if hasattr(opts, 'num_groups') else opts.num_worker
                delta_val = opts.delta
                sigma_val = opts.sigma

                estimated_B = opts.B if not opts.FedPG_BR else opts.Bmax  # worst-case for deterministic evasion
                V = 2 * np.log(2 * world_k / delta_val)
                # 保守估计 0.9，确保绝对安全
                tau_estimate = 2 * sigma_val * math.sqrt(V / estimated_B) * 0.9

                # 计算恶意梯度和诚实梯度的 L2 范数
                m_norm_sq = sum(torch.sum(m_g ** 2).item() for m_g in malicious_grad)
                m_norm = math.sqrt(m_norm_sq) + 1e-10
                honest_norm_sq = sum(torch.sum(h_g ** 2).item() for h_g in honest_grad)
                honest_norm = math.sqrt(honest_norm_sq) + 1e-10

                # Closed-form optimal injection: λ_max = τ / ||g_syn||
                optimal_lambda = tau_estimate / m_norm

                # Cap: keep combined gradient within 2x honest norm to avoid
                # outlier detection (||g_syn|| can be tiny on decision boundaries,
                # causing λ* to explode even though τ constraint is satisfied).
                max_lambda = 2.0 * honest_norm / m_norm
                if optimal_lambda > max_lambda:
                    optimal_lambda = max_lambda

                # g_combined = g_honest + λ · g_syn
                grad = [h_g + optimal_lambda * m_g for h_g, m_g in zip(honest_grad, malicious_grad)]

                # ---- DIAGNOSTIC: print once per epoch for the first Byzantine ----
                if not hasattr(self, '_diag_printed'):
                    self._diag_printed = False
                if not self._diag_printed:
                    honest_norm = math.sqrt(sum(torch.sum(h**2).item() for h in honest_grad))
                    combined_norm = math.sqrt(sum(torch.sum(g**2).item() for g in grad))
                    capped = " (capped)" if optimal_lambda < tau_estimate / (m_norm + 1e-10) else ""
                    print(f"[BSA-DIAG] worker {self.id} (group {self.group_id}): "
                          f"||g_honest||={honest_norm:.4e}, "
                          f"||g_syn||={m_norm:.4e}, "
                          f"tau_est={tau_estimate:.4e}, "
                          f"lambda*={optimal_lambda:.4e}{capped}, "
                          f"||g_combined||={combined_norm:.4e}, "
                          f"ratio={combined_norm/(honest_norm+1e-10):.2f}, "
                          f"triggers={len(trigger_data)}/{len(batch_states)}")
                    self._diag_printed = True

            else:
                grad = [item.grad for item in self.parameters()]
        # ---- End DBSA Attack ----

        # determine if the agent is byzantine (other attack types)
        elif self.is_Byzantine and self.attack_type is not None:
            # return wrong gradient with noise
            grad = []
            for item in self.parameters():
                if self.attack_type == 'zero-gradient':
                    grad.append(item.grad * 0)

                elif self.attack_type == 'random-noise':
                    rnd = (torch.rand(item.grad.shape, device = item.device) * 2 - 1) * (item.grad.max().data - item.grad.min().data) * 3
                    grad.append(item.grad + rnd)

                elif self.attack_type == 'sign-flipping':
                    grad.append(-2.5 * item.grad)

                elif self.attack_type == 'reward-flipping':
                    grad.append(item.grad)
                    # refer to collect_experience_for_training() to see attack

                elif self.attack_type == 'random-action':
                    grad.append(item.grad)
                    # refer to collect_experience_for_training() to see attack

                elif self.attack_type == 'random-reward':
                    grad.append(item.grad)
                    # refer to collect_experience_for_training() to see attack

                elif self.attack_type == 'FedScsPG-attack':
                    grad.append(item.grad)
                    # refer to agent.py to see attack

                elif self.attack_type == 'normalized-attack':
                    # Normalized Attack (WWW 2025):
                    # Byzantine workers initially return honest gradients.
                    # The server then manipulates these gradients in two stages:
                    #   Stage I  - optimize direction (lambda)
                    #   Stage II - optimize magnitude (zeta)
                    # See agent.py for the full attack implementation.
                    grad.append(item.grad)

                else: raise NotImplementedError()

        else:
            # return true gradient
            grad = [item.grad for item in self.parameters()]

        # report the results to the agent for training purpose
        return grad, batch_loss.item(), np.mean(batch_rets), np.mean(batch_lens)


    def to(self, device):
        self.logits_net.to(device)
        return self
    
    def eval(self):
        self.logits_net.eval()
        return self
        
    def train(self):
        self.logits_net.train()
        return self
    
    def parameters(self):
        return self.logits_net.parameters()
