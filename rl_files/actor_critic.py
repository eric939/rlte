import os
import random
from dataclasses import dataclass
import time
from typing import Optional
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.distributions.dirichlet import Dirichlet
from torch.utils.tensorboard import SummaryWriter
# sys path hacks 
import sys 
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(current_dir)
from simulation.market_gym import Market 

@dataclass
class Args:
    # other options dirichlet, normal
    exp_name: str = 'log_normal'
    # exp_name: str = 'dirichlet'
    # exp_name: str = 'log_normal_learn_std'
    # exp_name: str = 'dirichlet'
    tag: Optional[str] = None
    """additional tag for the experiment, should be string type"""
    seed: int = 0
    """seed of the experiment"""
    eval_seed: int = 100
    """seed for evaluation"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    save_model: bool = True
    """whether to save model """
    evaluate: bool = True
    """whether to evaluate the model"""
    n_eval_episodes: int = int(1e4)
    n_eval_episodes: int = 10
    """the number of episodes to evaluate the model"""
    run_directory: str = 'runs'
    """directory for saving models"""
    run_name:  Optional[str] = None 
    """to be filled at runtime, should be string type"""

    # Algorithm specific arguments
    drop_feature: Optional[str] = 'drift'
    '''feature to be dropped from observation, can be "volume", "order_info", "drift", "None" or None'''
    env_type: str = "strategic"
    # noise, flow, strategic 
    """the id of the environment"""
    num_lots: int = 20
    num_lots: int = 40
    """the number of lots"""
    terminal_time: int = 150
    # terminal_time: int = 300
    """the terminal time for the execution agent"""
    time_delta: int = 15
    # this setting leads to 10 time steps. num of lots should be divisuble by 10
    """the time delta for the execution agent"""
    # total_timesteps: int = 200*128*100
    # total_timesteps = itertaions * n_cpus * n_steps (in each evnironment)
    # 10 iterations is one full episode 
    total_timesteps: int = 200*128*100
    total_timesteps: int = 50
    # debug 
    # total_timesteps: int = 2*10
    # total_timesteps: int = 500*128*100
    """total timesteps of the experiments"""
    learning_rate: float = 5e-4
    """the learning rate of the optimizer"""
    # num_envs: int = 128
    num_envs: int = 1
    # num_envs: int = 1
    # num_envs: int = 1 
    """the number of parallel game environments"""
    # num_steps: int = 100
    num_steps: int = 10
    # num_steps: int = 10
    # less value bootstraping --> user more steps per environment 
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 1.0
    """the discount factor gamma"""
    # do additional test here with gae_lambda = 0.95
    gae_lambda: float = 1.0 
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 1
    """the number of mini-batches"""
    update_epochs: int = 1
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.5
    """the surrogate clipping coefficient"""
    clip_vloss: bool = False
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

def make_env(config):
    def thunk():
        return Market(config)
    return thunk
 
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs):
        n_hidden_units = 128 
        super().__init__()
        # critic network with 2 hidden layers
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), n_hidden_units)),
            nn.Tanh(),
            layer_init(nn.Linear(n_hidden_units, n_hidden_units)),
            nn.Tanh(),
            layer_init(nn.Linear(n_hidden_units, 1), std=1.0),
        )
        # action network with 2 hidden layers
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), n_hidden_units)),
            nn.Tanh(),
            layer_init(nn.Linear(n_hidden_units, n_hidden_units)),
            nn.Tanh(),
            # this is different than the logistic normal agent, no -1 here 
            layer_init(nn.Linear(n_hidden_units, np.prod(envs.single_action_space.shape)), std=1e-5),
        )
        # still use the same bias logic in the last layer [-1,-1, ... , -1, 1]
        x = -1.0*torch.ones(np.prod(envs.single_action_space.shape))
        x[-1] = 1.0
        self.actor_mean[-1].bias.data.copy_(x)
        # variance is scaled manually during training
        self.variance = 1.0 
        
    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)        
        action_std = torch.ones_like(action_mean)*self.variance
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

class AgentLogisticNormal(nn.Module):
    def __init__(self, envs, variance_scaling=True):
        n_hidden_units = 128 
        super().__init__()
        # critic network with 2 hidden layers
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), n_hidden_units)),
            nn.Tanh(),
            layer_init(nn.Linear(n_hidden_units, n_hidden_units)),
            nn.Tanh(),
            layer_init(nn.Linear(n_hidden_units, 1), std=1.0),
        )
        # action network with 2 hidden layers
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), n_hidden_units)),
            nn.Tanh(),
            layer_init(nn.Linear(n_hidden_units, n_hidden_units)),
            nn.Tanh(),
            layer_init(nn.Linear(n_hidden_units, np.prod(envs.single_action_space.shape)-1), std=1e-5),
        )
        # custom bias in the last layer [-1,-1, ... , -1, 1]
        x = -1.0*torch.ones(np.prod(envs.single_action_space.shape)-1)
        x[-1] = 1.0
        self.actor_mean[-1].bias.data.copy_(x)

        # variance is scaled manually during training
        self.variance_scaling = variance_scaling
        if variance_scaling: 
            self.variance = 1.0 
        else:
            self.variance = None 
            self.log_std = nn.Parameter(torch.zeros(np.prod(envs.single_action_space.shape)-1), requires_grad=True)
        # self.log_std = nn.Parameter(torch.zeros(np.prod(envs.single_action_space.shape)-1), requires_grad=True)

        # additional experiment no variance sclaing. but learning the std as a parameter


    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None, deterministic=False):
        action_mean = self.actor_mean(x)        
        if self.variance_scaling:
            action_std = torch.ones_like(action_mean)*self.variance
        else:   
            action_std = self.log_std.expand_as(action_mean)
            action_std = torch.exp(action_std)
        probs = Normal(action_mean, action_std)
        with torch.no_grad():
            if action is None:
                # sample base action, then apply logistic transformation, a = h(v)
                base_action = probs.sample()
                z = 1 + torch.sum(torch.exp(base_action), dim=1, keepdim=True)
                action = torch.exp(base_action)/z
                action = torch.cat((action, 1/z), dim=1)
            else:
                # use inverse logistic transform to get the base action v = h^{-1}(a)
                last_component = action[:,-1].reshape(-1,1)
                base_action = torch.log(action[:,:-1]/last_component)
        return action, probs.log_prob(base_action).sum(1), probs.entropy().sum(1), self.critic(x)

    def deterministic_action(self, x):
        # 
        # use mean base action, then apply logistic transformation, a = h(v)
        action_mean = self.actor_mean(x)        
        with torch.no_grad():
            # final_var = 0.32**2
            # base_action = action_mean + 0.5*final_var
            ## ADDING FINAL VARIANCE. SEE BELOW 
            base_action = action_mean
            z = 1 + torch.sum(torch.exp(base_action), dim=1, keepdim=True)
            action = torch.exp(base_action)/z
            action = torch.cat((action, 1/z), dim=1)
        return action
        

class DirichletAgent(nn.Module):
    def __init__(self, envs):         
        n_hidden_units = 128
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), n_hidden_units)),
            nn.Tanh(),
            layer_init(nn.Linear(n_hidden_units, n_hidden_units)),
            nn.Tanh(),
            layer_init(nn.Linear(n_hidden_units, 1), std=1.0),
        )    
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), n_hidden_units)),
            nn.Tanh(),
            layer_init(nn.Linear(n_hidden_units, n_hidden_units)),
            nn.Tanh(),
            # the last term scales the weights ! 
            layer_init(nn.Linear(n_hidden_units, np.prod(envs.single_action_space.shape)), std=1e-5),
        )

        # custom bias such that output after activation is [1,1,...,1,10]
        x = torch.log(torch.exp(torch.ones(np.prod(envs.single_action_space.shape)))-1)
        x[-1] = torch.log(torch.exp(torch.tensor(10.0))-1) 
        self.actor_mean[-1].bias.data.copy_(x)
        # print('Dirichlet agent initialized with bias ', self.actor_mean[-1].bias.data)
        # print(f'bias after softplus activation: {torch.nn.functional.softplus(self.actor_mean[-1].bias.data)}')
        # self.actor_var_scale = nn.Parameter(torch.tensor(-1.0), requires_grad=True)
        # self.actor_var_scale = 1e-1
    
    def get_action_and_value(self, state, action=None): 
        mean = torch.nn.functional.softplus(self.actor_mean(state))  
        # print('Mean after softplus activation:', mean[0].detach().cpu().numpy())
        # scale = torch.nn.functional.softplus(self.actor_var_scale) + 1e-5
        # scale = 1e-5
        if torch.isnan(state).any():
            print("State contains NaN", state)
        if torch.isnan(mean).any():
            print("Mean contains Nan, state is ", mean)
        # if torch.isnan(scale).any():
        #     print("Scale contains NaNs:", scale)
        # scale =x 1e-5
        # concentrations = mean*scale 
        # concentrations = mean
        # concentrations = mean*sclae
        # probs = Dirichlet(mean)
        probs = Dirichlet(mean)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.variance, self.critic(state)

    def get_value(self, state):
        return self.critic(state)
    
    def deterministic_action(self, state):
        # get concentration parameters from the network
        mean = torch.nn.functional.softplus(self.actor_mean(state))  
        return mean / torch.sum(mean, dim=1, keepdim=True)
    

if __name__ == "__main__":
    args = tyro.cli(Args)
    print('starting the training process')
    print(f'environment set up: volume={args.num_lots}, market_env={args.env_type}')
    
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    print('\n-----')
    print(f'batch_size={args.batch_size}, minibatch_size={args.minibatch_size}, num_iterations={args.num_iterations}, learning_rate={args.learning_rate}, num_iterations={args.num_iterations}, num_envs={args.num_envs}, num_steps_per_env={args.num_steps}, n_evalutation_episodes={args.n_eval_episodes}')
    print('-----')
    print('\n')
    print(f'time_delta={args.time_delta}, terminal_time={args.terminal_time}, lots={args.num_lots}')
    print('-----')

    # information should include: env_type, num_lots, seed, num_iterations, batch_size, algo_name
    # algo_name should describe the name of the algorithm, like log normal, dirichlet, normal softmax 
    # note that we are always using the actor critic algorithm, so we do not need to mention this 
    # naming convention: 
    assert args.drop_feature in ['volume', 'order_info', 'drift', 'None', None], f'unknown drop_feature {args.drop_feature}'
    
    if args.drop_feature == 'None':
        args.drop_feature = None

    feature_info = f'_{args.drop_feature}' if args.drop_feature is not None else ''
    if args.tag:
        print(f'additional tag for the experiment: {args.tag}')
        run_name = f"{args.env_type}_{args.num_lots}_seed_{args.seed}_eval_seed_{args.eval_seed}_eval_episodes_{args.n_eval_episodes}_num_iterations_{args.num_iterations}_bsize_{args.batch_size}_{args.exp_name}_{args.tag}{feature_info}"
        if args.tag == 'GAE':
            args.gae_lambda = 0.95
            print(f'using GAE with lambda = {args.gae_lambda}')
    else:
        print('no additional tag for the experiment')
        run_name = f"{args.env_type}_{args.num_lots}_seed_{args.seed}_eval_seed_{args.eval_seed}_eval_episodes_{args.n_eval_episodes}_num_iterations_{args.num_iterations}_bsize_{args.batch_size}_{args.exp_name}{feature_info}"    
    print(f'the run name is: {run_name}')
    args.run_name = run_name

    summary_path = f"{parent_dir}/tensorboard_logs/{run_name}"
    print(f'writing summary to {summary_path}:')
    writer = SummaryWriter(summary_path)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # device = torch.device("cuda:1" if torch.cuda.is_available() and args.cuda else "cpu")
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")
    if num_gpus > 0:
        # Choose the last GPU
        last_gpu = num_gpus - 1
        device = torch.device(f"cuda:{last_gpu}")
        print(f"Using GPU: {torch.cuda.get_device_name(last_gpu)} (cuda:{last_gpu})")
    else:
        # Fall back to CPU if no GPU is available
        device = torch.device("cpu")
        print("No GPU available, using CPU.")

    # TODO: additional option in config to remove features
    # This will then have an effect on how observations are generated  
    # include as option in config and add to name 
    # remove_featuer: "drift", "orders_info", "book_volumes"
    # then add this to the saving name 

    # environment setup
    print(f'dropping feature: {args.drop_feature}') if args.drop_feature is not None else print('not dropping any feature')
    configs = [{'market_env': args.env_type , 'execution_agent': 'rl_agent', 'volume': args.num_lots, 'seed': args.seed+s, 
                'terminal_time': args.terminal_time, 'time_delta': args.time_delta, 'drop_feature': args.drop_feature} for s in range(args.num_envs)]
    if args.exp_name == 'normal':
        # if we use just normal distribution, let the environment transform actions from R^n to the simplex 
        configs = [{'market_env': args.env_type , 'execution_agent': 'rl_agent', 'volume': args.num_lots, 'seed': args.seed+s, 
                    'terminal_time': args.terminal_time, 'time_delta': args.time_delta, 'transform_action': True, 'drop_feature': args.drop_feature} for s in range(args.num_envs)]
    env_fns = [make_env(config) for config in configs]
    envs = gym.vector.AsyncVectorEnv(env_fns=env_fns)
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    observation, info = envs.reset(seed=args.seed)
    print(f'observation space: {envs.single_observation_space}, action space: {envs.single_action_space}')

    # agent set up. we have three cases log_normal, dirichlet, and normal 
    if args.exp_name == 'log_normal':
        agent = AgentLogisticNormal(envs).to(device)
    elif args.exp_name == 'log_normal_learn_std':
        agent = AgentLogisticNormal(envs, variance_scaling=False).to(device)
    elif args.exp_name == 'dirichlet':
        agent = DirichletAgent(envs).to(device)
    elif args.exp_name == 'normal':
        agent = Agent(envs).to(device)
    else:
        raise ValueError(f"unknown agent type: {args.exp_name}")
    print(f'the agent type is: {args.exp_name}')
    # print(f'the agent is: {agent}')
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # start the simulation 
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device) 
    next_done = torch.zeros(args.num_envs).to(device)

    if args.num_iterations < 2: 
        raise ValueError('num_iterations should be greater than 1')

    for iteration in range(0, args.num_iterations):
        print(f'iteration={iteration}')
        # Annealing the rate if instructed to do so.
        returns = []
        times = []
        drifts = []
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
            print(f' the lerning rate is {lrnow}')        
        # manual standard deviation scalig. updated this to 0.1
        if args.exp_name == 'log_normal' or args.exp_name == 'normal':
            agent.variance = (0.32-1)*(iteration)/(args.num_iterations-1) + 1        
        # dirichlet agent does not use variance scaling 
        # agent.variance = 1 - iteration/(args.num_iterations+1) + 5e-1
        # keep same variance throughout the training
        # agent.variance = 1.0
        # if args.exp_name == 'normal' or args.exp_name == 'log_normal':            
            # print(f'the current variance is {agent.variance}')
        
        # this is the data collection loop 
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info is not None:
                        returns.append(info['reward'])
                        times.append(info['time'])
                        drifts.append(info['drift'])
        
        writer.add_scalar("charts/return", np.mean(returns), global_step)
        writer.add_scalar("charts/time", np.mean(times), global_step)
        writer.add_scalar("charts/drift", np.mean(drifts), global_step)        
            
        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                # could remove gamma and lambda if this is 1 anyways 
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            # shuffle indices could be removed since we are doing only one epoch
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                # log probs are computed with old actions 
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss modified 
                pg_loss = -mb_advantages * newlogprob
                pg_loss = pg_loss.mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                # does args.vf_coef need to be tuned ? 
                # could add entropy loss here 
                entropy_loss = entropy.mean()
                loss = pg_loss  + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/total_loss", loss, global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        if args.exp_name == 'log_normal' or args.exp_name == 'normal':
            writer.add_scalar("values/variance", agent.variance, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.save_model:
        model_path = f"{parent_dir}/models/{run_name}.pt"        
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")
    
    if args.evaluate:
        # deterministic_actions 
        print('\n starting evaluation')
        print('using deterministic actions for evaluation')
        # tag = 'deterministic_action'
        configs = [{'market_env': args.env_type , 'execution_agent': 'rl_agent', 'volume': args.num_lots, 
                    'seed': args.eval_seed+s, 'terminal_time': args.terminal_time, 'time_delta': args.time_delta, 'drop_feature': args.drop_feature} for s in range(args.num_envs)]
        if args.exp_name == 'normal':
            configs = [{'market_env': args.env_type , 'execution_agent': 'rl_agent', 'volume': args.num_lots, 
                        'seed': args.eval_seed+s, 'terminal_time': args.terminal_time, 'time_delta': args.time_delta, 'transform_action': True, 'drop_feature': args.drop_feature} for s in range(args.num_envs)]
        print('evalutation config:')
        print(configs[0])
        env_fns = [make_env(config) for config in configs]
        envs = gym.vector.AsyncVectorEnv(env_fns=env_fns)
        print('evalutation environment is created')
        obs, _ = envs.reset()
        episodic_returns = []
        start_time = time.time()
        while len(episodic_returns) < args.n_eval_episodes:
            with torch.no_grad():
                # always use deterministic action for evaluation
                # if tag == 'deterministic_action':
                actions = agent.deterministic_action(torch.Tensor(obs).to(device))
                # else:
                # actions, _, _, _ = agent.get_action_and_value(torch.Tensor(obs).to(device))
                next_obs, _, _, _, infos = envs.step(actions.cpu().numpy())
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info is not None:
                        episodic_returns.append(info['reward'])
            obs = next_obs
        print(f'evaluation time: {time.time()-start_time}')
        print(f'reward length: {len(episodic_returns)}')
        rewards = np.array(episodic_returns)        
        assert args.run_name is not None, "run_name should be set"
        # use tags such as long_horizon, GAE, or whatever
        file_name = f'{parent_dir}/rewards/{args.run_name}.npz'
        print(f'save rewards to {file_name}')
        np.savez(file_name, rewards=rewards)
        # if args.tag is not None:
        #     file_name = f'{parent_dir}/rewards/{args.run_name}_{args.tag}.npz'
        # else:

    envs.close()
    writer.close()



