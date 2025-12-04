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
num_envs = 4
num_envs = 128
num_steps = 10
time_delta = 15
terminal_time = 150
exp_name = 'dirichlet'
exp_name = 'log_normal'
batch_size = 12800
num_iterations = 400
n_eval_episodes = 10000

tag = 'deterministic_action'

for env_type in ['noise', 'flow', 'strategic']:
    for num_lots in [20, 60]:
        print('\n---')
        print(f'volume curves for env: {env_type}, volume: {num_lots}')
        run_name = (
            f"{env_type}_{num_lots}_seed_{train_seed}_eval_seed_{eval_seed}_"
            f"eval_episodes_{n_eval_episodes}_num_iterations_{num_iterations}_"
            f"bsize_{batch_size}_{exp_name}"
        )
        model_path = f'{parent_dir}/models/{run_name}.pt'
        configs = [
            {
                'market_env': env_type,
                'execution_agent': 'rl_agent',
                'volume': num_lots,
                'seed': eval_seed + s,
                'terminal_time': terminal_time,
                'time_delta': time_delta,
                'drop_feature': None,
            }
            for s in range(num_envs)
        ]
        env_fns = [make_env(config) for config in configs]
        envs = gym.vector.AsyncVectorEnv(env_fns=env_fns)

        agent = AgentLogisticNormal(envs) if exp_name == 'log_normal' else DirichletAgent(envs)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        state_dict = torch.load(model_path, map_location=device)
        agent.load_state_dict(state_dict)
        agent.to(device)
        agent.eval()

        obs, infos = envs.reset()
        N = 1000
        volume_buffers = [[] for _ in range(num_envs)]
        for idx in range(num_envs):
            volume_buffers[idx].append(infos['volume'][idx])
        completed_curves = []
        start_time = time.time()

        while len(completed_curves) < N:
            with torch.no_grad():
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
                if tag == 'deterministic_action':
                    actions = agent.deterministic_action(obs_tensor)
                else:
                    actions, _, _, _ = agent.get_action_and_value(obs_tensor)
                actions_np = actions.detach().cpu().numpy()
            next_obs, _, _, _, infos = envs.step(actions_np)

            volumes = infos['volume']
            final_infos = infos.get('final_info')
            if final_infos is None:
                final_infos = [None] * num_envs
            for idx in range(num_envs):
                final_info = final_infos[idx]
                if final_info is not None:
                    terminal_volume = final_info.get('volume', volumes[idx])
                    volume_buffers[idx].append(terminal_volume)
                    curve = np.array(volume_buffers[idx], dtype=np.float32)
                    if curve.shape[0] < num_steps + 1:
                        curve = np.pad(
                            curve,
                            pad_width=(0, num_steps + 1 - curve.shape[0]),
                            mode='constant',
                            constant_values=0,
                        )
                    elif curve.shape[0] > num_steps + 1:
                        raise ValueError(f'Curve length {curve.shape[0]} exceeds expected {num_steps + 1}')
                    completed_curves.append(curve)
                    if len(completed_curves) >= N:
                        volume_buffers[idx] = []
                        continue
                    volume_buffers[idx] = [volumes[idx]]
                else:
                    volume_buffers[idx].append(volumes[idx])
            obs = next_obs

        print(f'evaluation time: {time.time()-start_time:.2f} seconds')
        volume_curves = np.vstack(completed_curves)
        volume_curves = volume_curves[:N,:]
        assert volume_curves.shape == (N, num_steps + 1), f'shape is {volume_curves.shape}'
        assert np.all(volume_curves[:, 0] == num_lots)
        assert np.all(volume_curves[:, -1] == 0)
        if tag:
            np.savez(
                f'{parent_dir}/volume_curves/{env_type}_{num_lots}_samples_{N}_{exp_name}_{tag}.npz',
                volume_curves=volume_curves,
            )
        else:
            np.savez(
                f'{parent_dir}/volume_curves/{env_type}_{num_lots}_samples_{N}_{exp_name}.npz',
                volume_curves=volume_curves,
            )
