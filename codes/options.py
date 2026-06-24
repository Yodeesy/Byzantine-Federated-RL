#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import argparse
import torch

def get_options(args=None):
    parser = argparse.ArgumentParser('FT-FedScsPG')

    ### Overall run settings
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v1', choices = ['HalfCheetah-v2', 'LunarLander-v2', 'CartPole-v1'], 
                        help='OpenAI Gym env name for test')
    parser.add_argument('--eval_only', action='store_true', 
                        help='used only if to evaluate a pre-trained model')
    parser.add_argument('--no_saving', action='store_true', 
                        help='Disable saving checkpoints')
    parser.add_argument('--no_tb', action='store_true', 
                        help='Disable Tensorboard logging')
    parser.add_argument('--render', action='store_true', 
                        help='render to view the env')
    parser.add_argument('--mode', type=str, choices = ['human', 'rgb'], default='human', 
                        help='render mode')
    parser.add_argument('--log_dir', default = 'logs', 
                        help='log folder' )
    parser.add_argument('--run_name', default='run_name', 
                        help='Name to identify the experiments')
    
    
    # Multiple runs
    parser.add_argument('--multiple_run', type=int, default=1, 
                        help='number of repeated runs')
    parser.add_argument('--seed', type=int, default=0, 
                        help='Starting point of random seed when running multiple times')

    
    # Federation and Byzantine parameters
    parser.add_argument('--num_worker', type=int, default=10, 
                        help = 'number of worker node')
    parser.add_argument('--num_Byzantine', type=int, default=0, 
                        help = 'number of worker node that is Byzantine')
    parser.add_argument('--alpha', type=float, default=0.4, 
                        help = 'atmost alpha-fractional worker nodes are Byzantine')
    parser.add_argument('--attack_type', type=str, default='filtering-attack',
                        choices = ['zero-gradient', 'random-action', 'sign-flipping', 'reward-flipping', 'random-reward', 'random-noise', 'FedScsPG-attack', 'normalized-attack', 'divergence-attack'],
                        help = 'the behavior scheme of a Byzantine worker')

    # Ensemble defense (WWW 2025)
    parser.add_argument('--ensemble', action='store_true', default=False,
                        help='Enable ensemble defense (K groups, each trains independent global policy)')
    parser.add_argument('--num_groups', '-K', type=int, default=5,
                        help='Number of groups for ensemble defense')

    # Secure Aggregation (SecAgg)
    parser.add_argument('--use_secagg', action='store_true', default=False,
                        help='Enable secure aggregation. Server sees only '
                             'masked gradients; FedPG-BR filtering disabled.')

    # Override FedPG-BR / Normalized Attack hyperparams (for reproduction tuning)
    parser.add_argument('--sigma', type=float, default=None,
                        help='Override FedPG-BR variance bound sigma (default: env-specific)')
    parser.add_argument('--lambda_hat', type=float, default=None,
                        help='Override Normalized Attack lambda_hat (default: env-specific)')
    parser.add_argument('--zeta_hat', type=float, default=None,
                        help='Override Normalized Attack zeta_hat (default: env-specific)')

    # Divergence Attack (Cross-Group)
    parser.add_argument('--entropy_threshold', type=float, default=None,
                        help='Entropy threshold for divergence attack trigger states '
                             '(default: 0.6 * log(n_actions))')
    parser.add_argument('--target_action', type=int, default=0,
                        help='Target action Byzantine workers push toward in divergence attack')
    parser.add_argument('--target_action_mode', type=str,
                        choices=['fixed', 'group-mod'], default='fixed',
                        help='"fixed": all Byzantine push --target_action; '
                             '"group-mod": each Byzantine pushes group_id %% n_actions')
        
    
    # RL Algorithms (default GOMDP)
    parser.add_argument('--SVRPG', action='store_true', 
                        help='run SVRPG')
    parser.add_argument('--FedPG_BR', action='store_true', 
                        help='run FT-FedScsPG')

    
    # Training and validating
    parser.add_argument('--val_size', type=int, default=10, 
                        help='Number of episoid used for reporting validation performance')
    parser.add_argument('--val_max_steps', type=int, default=1000, 
                        help='Maximum trajectory length used for reporting validation performance')
    

    # Load pre-trained modelss
    parser.add_argument('--load_path', default = None,
                        help='Path to load pre-trained model parameters')


    ### end of parameters
    opts = parser.parse_args(args)

    # Save CLI overrides before env-specific defaults overwrite them
    _sigma_override = opts.sigma
    _lambda_hat_override = opts.lambda_hat
    _zeta_hat_override = opts.zeta_hat

    opts.use_cuda = False
    opts.run_name = "{}_{}".format(opts.run_name, time.strftime("%Y%m%dT%H%M%S"))
    opts.save_dir = os.path.join(
        'outputs',
        '{}'.format(opts.env_name),
        "worker{}_byzantine{}_{}".format(opts.num_worker, opts.num_Byzantine, opts.attack_type),
        opts.run_name
    ) if not opts.no_saving else None
    opts.log_dir = os.path.join(
        f'{opts.log_dir}',
        '{}'.format(opts.env_name),
        "worker{}_byzantine{}_{}".format(opts.num_worker, opts.num_Byzantine, opts.attack_type),
        opts.run_name
    ) if not opts.no_tb else None
    
    if opts.env_name == 'CartPole-v1':
        # Task-Specified Hyperparameters
        opts.max_epi_len = 500
        opts.max_trajectories = 5000
        opts.gamma  = 0.999
        opts.min_reward = 0  # for logging purpose (not important)
        opts.max_reward = 600  # for logging purpose (not important)

        # shared parameters
        opts.do_sample_for_training = True
        opts.lr_model = 1e-3
        opts.hidden_units = '16,16'
        opts.activation = 'ReLU'
        opts.output_activation = 'Tanh'
        
        # batch_size
        opts.B = 16 # for SVRPG and GOMDP
        opts.Bmin = 12 # for FT-FedScsPG
        opts.Bmax = 20 # for FT-FedScsPG
        opts.b = 4 # mini batch_size for SVRPG and FT-FedScsPG
        
        # inner loop iteration for SVRPG
        opts.N = 3
        
        # Filtering hyperparameters for FT-FedScsPG
        opts.delta = 0.6
        opts.sigma = 0.06

        # Normalized attack hyperparameters (WWW 2025, Table 4)
        # lambda_hat: initial step size for direction optimization (Stage I)
        # zeta_hat: initial step size for magnitude optimization (Stage II)
        # Both decay by factor 1/3 each iteration
        opts.lambda_hat = 0.83
        opts.zeta_hat = 0.03

        # Divergence attack entropy threshold (0.6 * ln(2))
        if opts.entropy_threshold is None:
            opts.entropy_threshold = 0.42


    elif opts.env_name == 'HalfCheetah-v2':
        # Task-Specified Hyperparameters
        opts.max_epi_len = 500  
        opts.max_trajectories = 1e4
        opts.gamma  = 0.995
        opts.min_reward = 0 # for logging purpose (not important)
        opts.max_reward = 4000 # for logging purpose (not important)
        
        # shared parameters
        opts.do_sample_for_training = True
        opts.lr_model = 8e-5 # 4e-3
        opts.hidden_units = '64,64'
        opts.activation = 'Tanh'
        opts.output_activation = 'Tanh'
       
        # batch_size
        opts.B = 48 # for SVRPG and GOMDP
        opts.Bmin = 46 # for FT-FedScsPG
        opts.Bmax = 50 # for FT-FedScsPG
        opts.b = 16 # mini batch_size for SVRPG and FT-FedScsPG
        
        # inner loop iteration for SVRPG
        opts.N = 3
    
        # Filtering hyperparameters for FT-FedScsPG
        opts.delta = 0.6
        opts.sigma = 0.9

        # Normalized attack hyperparameters (WWW 2025, Table 4)
        opts.lambda_hat = 0.83
        opts.zeta_hat = 0.2

        # Divergence attack (continuous: use variance-based trigger)
        if opts.entropy_threshold is None:
            opts.entropy_threshold = 0.5  # action variance threshold


    if opts.env_name == 'LunarLander-v2':
        # Task-Specified Hyperparameters
        opts.max_epi_len = 1000  
        opts.max_trajectories = 1e4
        opts.gamma  = 0.99
        opts.min_reward = -1000 # for logging purpose (not important)
        opts.max_reward = 300 # for logging purpose (not important)
        
        # shared parameters
        opts.do_sample_for_training = True
        opts.lr_model = 1e-3 # 8e-4
        opts.hidden_units = '64,64'
        opts.activation = 'Tanh'
        opts.output_activation = 'Tanh'
        
        # batch_size
        opts.B = 32 # for SVRPG and GOMDP
        opts.Bmin = 26 # for FT-FedScsPG
        opts.Bmax = 38 # for FT-FedScsPG
        opts.b = 8 # mini batch_size for SVRPG and FT-FedScsPG
        
        # inner loop iteration for SVRPG
        opts.N = 3
        
        # Filtering hyperparameters for FT-FedScsPG
        opts.delta = 0.6
        opts.sigma = 0.07

        # Normalized attack hyperparameters (WWW 2025, Table 4)
        opts.lambda_hat = 1.0
        opts.zeta_hat = 0.02

        # Divergence attack entropy threshold (0.6 * ln(4))
        if opts.entropy_threshold is None:
            opts.entropy_threshold = 0.83

    # Allow CLI override of key hyperparams for reproduction tuning
    if _sigma_override is not None:
        opts.sigma = _sigma_override
    if _lambda_hat_override is not None:
        opts.lambda_hat = _lambda_hat_override
    if _zeta_hat_override is not None:
        opts.zeta_hat = _zeta_hat_override

    assert opts.SVRPG + opts.FedPG_BR <= 1
    print('run GPMDP\n' if opts.SVRPG + opts.FedPG_BR == 0 else ('run FT-FedScsPG\n' if opts.FedPG_BR else 'run SVRPG\n'))

    return opts
