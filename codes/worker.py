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
        self.env = gym.make(env_name)
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

        # ---- Divergence Attack: local synthetic gradient on high-entropy states ----
        if self.is_Byzantine and self.attack_type == 'divergence-attack':
            from torch.distributions.categorical import Categorical
            from gymnasium.spaces import Discrete

            n_actions = (self.env.action_space.n
                         if isinstance(self.env.action_space, Discrete)
                         else self.env.action_space.shape[0])
            max_entropy = np.log(n_actions)
            threshold = (opts.entropy_threshold
                         if opts and opts.entropy_threshold is not None
                         else 0.6 * max_entropy)

            # Cross-Group targeting: each group pushes a DIFFERENT action
            if (opts and opts.target_action_mode == 'group-mod'
                    and hasattr(self, 'group_id') and self.group_id >= 0):
                target_action = self.group_id % n_actions
            else:
                target_action = opts.target_action if opts else 0

            # Scan visited states for high-entropy decision boundaries
            trigger_states = []
            for obs in batch_states:
                obs_t = torch.as_tensor(obs, dtype=torch.float32).to(device)
                logits = self.logits_net.logits_net(obs_t)
                dist = Categorical(logits=logits)
                if dist.entropy().item() > threshold:
                    trigger_states.append(obs_t)

            if len(trigger_states) > 0:
                # Save honest gradient
                honest_grad = [p.grad.clone() for p in self.parameters()]

                # Synthetic gradient: push toward target_action on trigger states
                trigger_tensor = torch.stack(trigger_states)
                logits_t = self.logits_net.logits_net(trigger_tensor)
                dist_t = Categorical(logits=logits_t)
                logp_target = dist_t.log_prob(
                    torch.tensor(target_action, device=device))
                loss_syn = -logp_target.mean()  # maximize logp of target action

                self.logits_net.zero_grad()
                loss_syn.backward()

                # Combine honest + synthetic, then norm-match for stealth
                beta = 1.0  # poisoning strength
                grad = []
                for h_g, p in zip(honest_grad, self.parameters()):
                    combined = h_g + beta * p.grad
                    # Stealth: match norm to original honest gradient
                    h_norm = h_g.norm() + 1e-10
                    combined = combined / (combined.norm() + 1e-10) * h_norm
                    grad.append(combined)
            else:
                # No trigger states found — return honest gradient
                grad = [item.grad for item in self.parameters()]
        # ---- End Divergence Attack ----

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
