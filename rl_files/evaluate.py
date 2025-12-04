from dataclasses import dataclass
from xml.parsers.expat import model
import gymnasium as gym
import torch
import numpy as np
import time
from simulation.market_gym import Market
from rl_files.actor_critic import AgentLogisticNormal, DirichletAgent
import os
parent_dir = os.path.abspath('.')

def make_env(config):
    def thunk():
        return Market(config)
    return thunk

# re-evaluate models 
# the benchmark models should be evaluates somehwere else 

env_type = 'noise'
num_lots = 20
train_seed = 0
eval_seed = 100
num_envs = 128 
time_delta = 15 
terminal_time = 150 
exp_name = 'log_normal' 
# 50*128 
batch_size = 6400 
num_iterations = 400
batch_size = 12800
# should eval again with more eval episodes 
n_eval_episodes = 10000

# if this is true, use deterministic action for evaluation
# feature importance with deterministic sampling 
deterministic_action = True 

## check new determnistic action implementation


# for exp_name in ['log_normal', 'dirichlet']:
# for exp_name in ['log_normal_order_info', 'log_normal_volume', 'log_normal_drift']:
# resample with deterministic action 
# for exp_name in ['log_normal_learn_std']:
for exp_name in ['log_normal']:
    for env_type in ['noise', 'flow', 'strategic']:
        for num_lots in [20, 60]:      
            print(f'evaluating {env_type} env, num_lots: {num_lots}, exp_name: {exp_name}')
            # this was the wrong number of eval episodes
            n_eval_episodes = 10000
            run_name = f"{env_type}_{num_lots}_seed_{train_seed}_eval_seed_{eval_seed}_eval_episodes_{n_eval_episodes}_num_iterations_{num_iterations}_bsize_{batch_size}_{exp_name}"    
            # path = '/u/weim/lob/models/noise_20_seed_0_eval_seed_100_eval_episodes_10_num_iterations_2_bsize_20_log_normal.pt'
            # print(run_name)
            model_path = f'{parent_dir}/models/{run_name}.pt'

            # which feature to drop             
            if 'order_info' in exp_name:
                drop_feature = 'order_info'
            elif 'volume' in exp_name:
                drop_feature = 'volume'
            elif 'drift' in exp_name:
                drop_feature = 'drift'
            else:
                drop_feature = None

            configs = [{'market_env': env_type , 'execution_agent': 'rl_agent', 'volume': num_lots, 'seed': eval_seed+s, 'terminal_time': terminal_time, 'time_delta': time_delta, 'drop_feature': drop_feature} for s in range(num_envs)]
            # if exp_name == 'normal':
            #     configs = [{'market_env': env_type , 'execution_agent': 'rl_agent', 'volume': num_lots, 'seed': eval_seed+s, 'terminal_time': terminal_time, 'time_delta': time_delta, 'transform_action': True} for s in range(num_envs)]
            # print('evalutation config:')
            # print(configs[0])
            env_fns = [make_env(config) for config in configs]
            envs = gym.vector.AsyncVectorEnv(env_fns=env_fns)

            # load the model
            # if 'learn_std' in exp_name:
            #     agent = AgentLogisticNormal(envs, variance_scaling=False)
            if exp_name.startswith('log_normal_learn_std'):
                agent = AgentLogisticNormal(envs, variance_scaling=False)
            elif exp_name.startswith('log_normal'):
                agent = AgentLogisticNormal(envs, variance_scaling=True)
            else:
                agent = DirichletAgent(envs)
            device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
            # print(device)
            state_dict = torch.load(model_path, map_location=device)
            # print(state_dict['critic.0.weight'].device)
            agent.load_state_dict(state_dict)
            agent.to(device)
            agent.eval()


            # print('evalutation environment is created')
            obs, _ = envs.reset()
            episodic_returns = []
            start_time = time.time()

            # print(agent.critic[0].bias.device)

            # check episodic returns here 
            n_eval_episodes = 10000
            while len(episodic_returns) < n_eval_episodes:
                with torch.no_grad():
                    obs = torch.Tensor(obs).to(device)  
                    # print(obs.device)
                    # actions, _, _, _ = agent.get_action_and_value(obs)
                    # this only works for logistic normal agent
                    if deterministic_action:
                        actions = agent.deterministic_action(obs)
                    else:
                        actions, _, _, _ = agent.get_action_and_value(obs)
                    next_obs, _, _, _, infos = envs.step(actions.cpu().numpy())
                if "final_info" in infos:
                    for info in infos["final_info"]:
                        if info is not None:
                            episodic_returns.append(info['reward'])
                obs = next_obs
            print(f'evaluation time: {time.time()-start_time}')
            print(f'reward length: {len(episodic_returns)}')
            rewards = np.array(episodic_returns)        
            assert run_name is not None, "run_name should be set"
            saving_tag = '_deterministic_action' if deterministic_action else ''
            file_name = f'{parent_dir}/rewards/{run_name}{saving_tag}_new.npz'
            np.savez(file_name, rewards=rewards)
            print(f'save rewards to {file_name}')