def rollout(seed, n_episodes, execution_agent, market_type, volume, terminal_time, time_delta):
    """
    rollout episodes for a market or limit order 

    Args:
        n_episodes: number of episodes to rollout
        order_type: market or limit 
        volume : number of lots to trade
        level: only for limit orders 
        market_type: noise, flow, or strategic 
        terminal_time: the time at which the simulation should terminate
    
    Returns:
        total_rewards: list of rewards for each episode
        times: list of terminal times for each episode
        n_events: list of number of events for each episode
    
    Raises:
        errors are raised within the Market Class, if the input configurations are wrong         
    """

    config = {'seed': seed, 'market_env': market_type, 'execution_agent': execution_agent, 'volume': volume, 'terminal_time': terminal_time, 'time_delta': time_delta}
    M = Market(config)
    total_rewards = []
    times = []
    n_events = []
    for _ in range(n_episodes):
        # M.reset() will run until the inventory is depletet for the benchmark agents 
        observation, info = M.reset()
        if execution_agent == 'rl_agent':
            # if an rl_agent is present, the environment runs until the next observation every time_delta 
            # it terminates if the execution agent's inventory is depleted
            print('NEW EPISODE')
            terminated = False
            while not terminated:
                # action depends on the dimension of the action space 
                action = np.array([0,0,1,0,0,0,0], dtype=np.float32)
                assert action in M.action_space
                observation, reward, terminated, truncated, info = M.step(action)
                assert observation in M.observation_space
        total_rewards.append(info['reward'])
        times.append(info['time'])
        n_events.append(info['n_events'])        
    return total_rewards, times, n_events