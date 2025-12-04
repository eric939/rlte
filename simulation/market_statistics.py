# TODO: should integrate this whole thing into market gym directly

from agents import NoiseAgent, LinearSubmitLeaveAgent, StrategicAgent, SubmitAndLeaveAgent, MarketAgent, RLAgent, InitialAgent, ObservationAgent, MarketAgent, LimitAgent
from limit_order_book.limit_order_book import LimitOrderBook, MarketOrder, LimitOrder 
from config.config import noise_agent_config, strategic_agent_config, sl_agent_config, linear_sl_agent_config, market_agent_config, initial_agent_config, observation_agent_config
import numpy as np
import pandas as pd 
from config.config import noise_agent_config
from queue import PriorityQueue
from dataclasses import dataclass, field
from typing import Any
from multiprocessing import Pool
import itertools
import time
import seaborn as sns
import sys
import matplotlib.pyplot as plt
import os 

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

"""
    - this code is desgigned to obtain market statistic: like mid price drift, number of trades, number of events
    - the statistics are averaged over episodes 
    - the prioity queue is set up manually (no universal class for this, just do case by case)
    - save plots in the plots folder and trades in the results folder
    - TODO: could write the code using the Market environment class: need to modify the Market class 
        - currently, the market class always requires an execution agent
        - would need to have the case with and without execution agent
"""


class Market():
    def __init__(self, market_env='noise', seed=0, send_limit=False, send_market=False, volume=10):
        assert market_env in ['noise', 'flow', 'strategic']
        self.agents = {}

        time_delta =15 
        # terminal_time = 150
        terminal_time = 150

        # initial agent setting 
        if market_env == 'noise':
            initial_agent_config['initial_shape_file'] = 'initial_shape/noise_65.npz'
            initial_agent_config['start_time'] = -time_delta
            agent = InitialAgent(**initial_agent_config)
            self.agents[agent.agent_id] = agent
        else:
            initial_agent_config['initial_shape_file'] = 'initial_shape/noise_flow_65.npz'
            initial_agent_config['start_time'] = -time_delta
            agent = InitialAgent(**initial_agent_config)
            self.agents[agent.agent_id] = agent
        agent = InitialAgent(**initial_agent_config)
        self.agents[agent.agent_id] = agent

        # noise agent setting
        noise_agent_config['rng'] = np.random.default_rng(seed)
        noise_agent_config['unit_volume'] = False
        noise_agent_config['terminal_time'] = terminal_time
        noise_agent_config['start_time'] = 0
        # noise_agent_config['fall_back_volume'] = 5

        # noise or flow 
        if market_env == 'noise':
            noise_agent_config['imbalance_reaction'] = False
            agent = NoiseAgent(**noise_agent_config)
        else: 
            noise_agent_config['imbalance_reaction'] = True
            agent = NoiseAgent(**noise_agent_config)
            # noise_agent_config['initial_shape_file'] = 'initial_shape/noise_flow_75_unit.npz'
            if market_env == 'strategic':
                # more scaling when strategic agent present
                agent.limit_intensities = agent.limit_intensities * 0.65
                agent.market_intensity = agent.market_intensity * 0.65
                agent.cancel_intensities = agent.cancel_intensities * 0.65
            else:                 
                agent.limit_intensities = agent.limit_intensities * 0.85
                agent.market_intensity = agent.market_intensity * 0.85
                agent.cancel_intensities = agent.cancel_intensities * 0.85
        self.agents[agent.agent_id] = agent
        
        # strategic agent setting 
        if market_env == 'strategic':
            strategic_agent_config['time_delta'] = 3
            strategic_agent_config['market_volume'] = 1
            strategic_agent_config['limit_volume'] = 2
            strategic_agent_config['rng'] = np.random.default_rng(seed)
            strategic_agent_config['terminal_time'] = terminal_time
            strategic_agent_config['start_time'] = 0
            agent = StrategicAgent(**strategic_agent_config)
            self.agents[agent.agent_id] = agent 
        
        # observation agent 
        observation_agent_config['start_time'] = 0
        observation_agent_config['terminal_time'] = terminal_time
        observation_agent_config['time_delta'] = 5 
        agent = ObservationAgent(**observation_agent_config)
        self.agents[agent.agent_id] = agent

        if send_market:
            MA = MarketAgent(volume=volume, start_time=0, priority=1)
            self.agents[MA.agent_id] = MA

        if send_limit:
            LA = LimitAgent(start_time=0, volume = volume, priority=1, level=1)
            self.agents[LA.agent_id] = LA

            
    def reset(self):
        list_of_agents = list(self.agents.keys()) 
        self.lob = LimitOrderBook(list_of_agents=list_of_agents, level=30, only_volumes=False)
        for agent_id in list_of_agents:
            self.agents[agent_id].reset()         
        # if 'strategic_agent' in self.agents:
        # self.agents['strategic_agent'].direction = 'sell'
        self.pq = PriorityQueue()
        for agent_id in self.agents:
            out = self.agents[agent_id].initial_event()
            self.pq.put(out)
        # add market agent
        # self.lob.registered_agents += ['market_agent']
        return None
    
    def run(self):
        n_events = 0 
        n_cancellations = 0 
        n_limits = 0 
        n_markets = 0 
        event = None
        mid_prices = []
        while not self.pq.empty(): 
            n_events += 1
            time, _, event = self.pq.get()
            orders = self.agents[event].generate_order(lob=self.lob, time=time)
            self.lob.process_order_list(orders) if orders is not None else None
            out = self.agents[event].new_event(time, event)
            if out is not None:
                self.pq.put(out)
            if event ==  'initial_agent':
                initial_mid = (self.lob.get_best_price('bid')+self.lob.get_best_price('ask'))/2
            # if market or limit order placement, then place the order here
            if event == 'observation_agent': 
                mid_prices.append((self.lob.get_best_price('bid')+self.lob.get_best_price('ask'))/2-initial_mid)
                # if time == 0: 
                #     # MO  = MarketOrder(agent_id='market_agent', side='bid', volume=60, time=time)
                #     # self.lob.process_order_list([MO])
                #     # limit order
                #     LO = LimitOrder(agent_id='market_agent', side='ask', volume=60, price=self.lob.get_best_price('ask'), time=time)
                #     self.lob.process_order_list([LO])
                #     # print('market order placed at time 0')

                    
                    
        drift = (self.lob.get_best_price('bid')+self.lob.get_best_price('ask'))/2 - initial_mid

        trades = np.sum(self.lob.data.market_buy)+np.sum(self.lob.data.market_sell)

        buy_orders = np.sum(self.lob.data.market_buy)

        sell_orders = np.sum(self.lob.data.market_sell)

        # add mid price list here
        return time, n_events, drift, trades, buy_orders, sell_orders, mid_prices


def rollout(seed, num_episodes, market_type, send_limit=False, send_market=False, volume=10):
    M = Market(market_env=market_type, seed=seed, send_limit=send_limit, send_market=send_market, volume=volume)
    n_events = []
    drifts = []
    times = []
    trades = []
    buy_orders = []
    sell_orders = []
    mid_price_list = []
    for _ in range(num_episodes):
        M.reset()
        time, n_event, drift, trade, buys, sells, mid_prices = M.run()
        n_events.append(n_event)
        drifts.append(drift)
        times.append(time)
        trades.append(trade)
        buy_orders.append(buys)
        sell_orders.append(sells)
        mid_price_list.append(mid_prices)
    return n_events, drifts, times, trades, buy_orders, sell_orders, mid_price_list


def mp_rollout(n_samples, n_cpus, market_type, send_limit=False, send_market=False, volume=10):
    """
        - TODO: do this with joblib
    """
    samples_per_env = int(n_samples/n_cpus) 
    with Pool(n_cpus) as p:
        out = p.starmap(rollout, [(seed, samples_per_env, market_type, send_limit, send_market, volume) for seed in range(n_cpus)])    
    n_events, drifts, times, trades, buys, sells, mid_prices = zip(*out)
    n_events = list(itertools.chain.from_iterable(n_events))
    times = list(itertools.chain.from_iterable(times))
    drifts = list(itertools.chain.from_iterable(drifts))
    trades = list(itertools.chain.from_iterable(trades))
    buys = list(itertools.chain.from_iterable(buys))
    sells = list(itertools.chain.from_iterable(sells))
    # Keep mid_prices as list of lists - each inner list contains mid prices for one run
    mid_prices = list(itertools.chain.from_iterable(mid_prices))
    return n_events, drifts, times, trades, buys, sells, mid_prices




if __name__ == '__main__':
    ## test 
    out = rollout(seed=0, num_episodes=10, market_type='strategic')
    print(len(out[-1]))

    # this section is for impact studies 
    if False: 
        # impact of limit orders for noise market 
        for market in ['noise', 'flow', 'strategic']:
            for exp in ['limit', 'market']:
                n_cpus = 100 
                n_samples = 1000
                # plt.figure(figsize=(10, 6))
                # exp = 'market' # 'limit'
                if exp == 'market':
                    send_market = True
                    send_limit = False
                else:
                    send_market = False
                    send_limit = True
                for volumes in [10, 20, 60]:
                    n_events, drifts, times, trades, buys, sells, mid_prices = mp_rollout(n_samples=n_samples, n_cpus=n_cpus, market_type=market, send_limit=send_limit, volume=volumes, send_market=send_market)
                    mid_prices = np.array(mid_prices)
                    # mid_prices = [np.array(prices) for prices in mid_prices]
                    # mid_prices = np.array(mid_prices)
                    print(mid_prices.shape)
                    # mid_prices = np.mean(mid_prices, axis=0)
                    terminal_time = 150
                    np.savez(f'{parent_dir}/market_impact_study/{market}_{exp}_vol{volumes}_{terminal_time}.npz', mid_prices=mid_prices)



    if True:
        envs = ['noise', 'flow', 'strategic']
        n_samples = 1000
        n_cpus = 128
        results = {f'n_events': [],'drift_mean': [], 'drift_std': [], 'trades': [], 'trades_std': [], 'buy_orders': [], 'sell_orders': []} 
        data_drifts = {}
        data_for_trade_plot = {}
        start_time = time.time()
        for env in envs:
            n_events, drifts, times, trades, buys, sells, mid_prices = mp_rollout(n_samples=n_samples, n_cpus=n_cpus, market_type=env)
            data_drifts[env] = drifts
            data_for_trade_plot[env] = trades
            results[f'n_events'].append(np.mean(n_events))
            results[f'drift_mean'].append(np.mean(drifts))
            results[f'drift_std'].append(np.std(drifts))
            results[f'trades'].append(np.mean(trades))
            results[f'trades_std'].append(np.std(trades))
            results[f'buy_orders'].append(np.mean(buys))
            results[f'sell_orders'].append(np.mean(sells))
        df = pd.DataFrame.from_dict(results).round(2)
        df = df.drop(columns=['drift_std', 'trades_std', 'trades_std'])
        df.index = envs
        print(df)
        df.to_latex(index=True, buf=f'{parent_dir}/latex_tables/market_stats.tex')


        if False:
            print(f'number of buys: {len(buys)}')
            print(len(mid_prices))
            print(mid_prices[-1])  
            end_time = time.time()
            execution_time = end_time - start_time
            print("Execution time:", execution_time, "seconds")
            mid_prices = [np.array(prices) for prices in mid_prices]
            print(f'length of mid prices list: {len(mid_prices)}')
            mid_prices_array = np.array(mid_prices)
            mid_prices_mean = np.mean(mid_prices_array, axis=0)
            mid_prices_std = np.std(mid_prices_array, axis=0)
            # print(mid_prices.shape)
            plt.figure(figsize=(10, 6))
            # plt.errorbar(range(len(mid_prices_mean)), mid_prices_mean, yerr=mid_prices_std, color='blue', linewidth=1, capsize=3, alpha=0.7)
            time_steps = np.arange(start=0, stop=155, step=5)  # Assuming mid_prices are recorded every 5 time units
            print(time_steps[-1])
            plt.plot(time_steps, mid_prices_mean, color='blue', linewidth=2)
            plt.xticks(range(0, 160, 10))
            plt.title(f'Average mid price after market order noise market', fontsize=16)
            plt.xlabel('Time in seconds')
            plt.ylabel('Mid price')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('plots/mid_price_evolution.png')
            plt.show()

    if False:
        # process results 
        results = pd.DataFrame.from_dict(results).round(2)
        results.index = envs 
        print(results)
        # results.to_csv(f'results/market_stats_std8.csv')

        # histogram of drifts 
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['blue', 'green']
        bins = np.arange(-4.25, 5.75, 0.5)
        ax.hist([data_drifts['noise'], data_drifts['flow']], bins, density=False, histtype='bar', color=colors, label=['Noise', 'Flow'], rwidth=0.8)
        ax.set_xticks(np.arange(-4, 4.5, 0.5))
        ax.tick_params(axis='x', labelsize=7)
        ax.legend(prop={'size': 10})
        ax.set_title('Mid Price Drift', fontsize=12)
        plt.grid(True)
        plt.xlim(-4, 4)
        # plt.tight_layout()
        plt.savefig('plots/mid_price_drift_std2_150.pdf')
        plt.figure(figsize=(10, 6))


        # plot histogram of trades for each market type
        plt.figure(figsize=(10, 6))
        sns.kdeplot(data_for_trade_plot['noise'], fill=False, label='Noise', clip=(0, 150))
        sns.kdeplot(data_for_trade_plot['flow'], fill=False, label='Flow',  clip=(0, 150))
        plt.legend(fontsize=12)
        plt.xlim(0, 140)
        plt.grid(True)
        plt.title('Number of Trades', fontsize=16)
        plt.ylabel('Frequency')
        # plt.tight_layout()
        plt.savefig('plots/kde_trades_std2_150.pdf')