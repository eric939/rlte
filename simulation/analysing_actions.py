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

env_type = 'noise'
num_lots = 20
train_seed = 0
eval_seed = 100
num_envs = 128
# num_envs = 4 
num_steps = 10 
time_delta = 15 
terminal_time = 150 
exp_name = 'log_normal' 
exp_name = 'dirichlet'
# 50*128 
batch_size = 12800  
num_iterations = 400
n_eval_episodes = 10000

# TODO: include the possibility to sample deterministic actions
# tag = 'deterministic_action'
tag = None 

for env_type in ['noise', 'flow', 'strategic']:
    for num_lots in [20, 60]:
        print('\n---')
        print(f'generating results for: {env_type}, volume: {num_lots}, exp_name: {exp_name}')
        run_name = f"{env_type}_{num_lots}_seed_{train_seed}_eval_seed_{eval_seed}_eval_episodes_{n_eval_episodes}_num_iterations_{num_iterations}_bsize_{batch_size}_{exp_name}"    
        model_path = f'{parent_dir}/models/{run_name}.pt'
        configs = [{'market_env': env_type , 'execution_agent': 'rl_agent', 'volume': num_lots, 'seed': eval_seed+s, 'terminal_time': terminal_time, 'time_delta': time_delta, 'drop_feature': None} for s in range(num_envs)]
        env_fns = [make_env(config) for config in configs]
        envs = gym.vector.AsyncVectorEnv(env_fns=env_fns)

        # load the model
        agent = AgentLogisticNormal(envs) if exp_name == 'log_normal' else DirichletAgent(envs)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        state_dict = torch.load(model_path, map_location=device)
        agent.load_state_dict(state_dict)
        agent.to(device)
        agent.eval()

        # print('evalutation environment is created')
        obs, infos = envs.reset()
        start_time = time.time()

        # check episodic returns here 
        N = 1000
        n_episodes = 0
        start_time = time.time()    
        start_trajectory = [True]*num_envs  # to record where trajectories start
        list_of_actions = []
        start_time = time.time()
        while len(list_of_actions) < N:
            with torch.no_grad():
                obs = torch.Tensor(obs).to(device)  
                # print(obs.device)
                if tag == 'deterministic_action':
                    actions = agent.deterministic_action(obs)
                else:
                    actions, _, _, _ = agent.get_action_and_value(obs)
                for idx in range(num_envs):
                    if start_trajectory[idx] and len(list_of_actions) < N:
                        list_of_actions.append(actions[idx, :].cpu().numpy())
                next_obs, _, _, _, infos = envs.step(actions.cpu().numpy())
                obs = next_obs
                # print(infos)
            if "final_info" in infos:
                # infos['final_info'] is a list with length num_envs
                # if there is no final info, the corresponding list entry is None
                x = num_envs*[False]
                for idx, info in enumerate(infos["final_info"]):
                    if info is not None:
                        n_episodes += 1
                        x[idx] = True
            else:
                x = num_envs*[False]
            start_trajectory = x 
                        # print(info['episodic_return'])
        print(f'Time taken to eval {N} episodes: {time.time()-start_time:.2f} seconds')
        
        actions = np.vstack(list_of_actions)  # shape (N, num_steps)
        if tag: 
            name = f'{parent_dir}/actions/{env_type}_{num_lots}_{exp_name}_{tag}.npz'
        else:
            name = f'{parent_dir}/actions/{env_type}_{num_lots}_{exp_name}.npz'
        print(f'Saving actions to {name}')
        np.savez(name, actions=actions)
        # save those actions and then plot them in a jupyter notebook



