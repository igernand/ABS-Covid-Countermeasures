import torch as T
from covid_abs.rlagent import RLAgent
import numpy as np
from matplotlib import animation, rc
from IPython.display import HTML

from covid_abs.graphics import  *
from covid_abs.network.graph_abs import *


def trainRL(rounds,load = True):   
    scenario = dict(
        name='scenario',
        policy = 'RL',
    )
    # build/load agent
    
    RLagent = RLAgent(gamma=0.7, epsilon=1.0, batch_size=64, n_actions=5, eps_end=0.01, input_dims=[5], lr=0.003)
    FILE = "modelinfo.pth"
    if load:
        checkpoint = T.load(FILE)
        RLagent.Q_eval.load_state_dict(checkpoint['model_state_dict'])
        RLagent.Q_eval.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        RLagent.epsilon = checkpoint['RLagent_epsilon']
        RLagent.state_memors = checkpoint['RLagent_state_memory']
        RLagent.new_state_memory = checkpoint['RLagent_new_state_memory']
        RLagent.action_memory = checkpoint['RLagent_action_memory']
        RLagent.reward_memory = checkpoint['RLagent_reward_memory']
        RLagent.terminal_memory = checkpoint['RLagent_terminal_memory']
        RLagent.mem_cntr = checkpoint['RLagent_mem_cntr']
    
    
    scores, eps_history = [], []
    n_rounds = rounds
    for j in range(n_rounds):
        # build simulation
        np.random.seed(1)
        sim = GraphSimulation(**{**scenario})
        from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                                   AutoMinorLocator)

        statistics = {'info': [], 'ecom': []}



        frames = 1440
        iteration_time = 25
        tick_unit = 72

        sim.initialize()
        score = 0
        done = False
        observation = sim.observation 
        for i in range(6): # act every 10 days
            df1, df2 = update_statistics(sim, statistics)

            tickslabels = [str(i//24) for i in range(0, frames, tick_unit)]
            
            action = RLagent.choose_action(observation)
            observation_, reward = sim.executeRL(action)

            score += reward
            if i==5: 
                done = True
            RLagent.store_transition(observation, action, reward,
									observation_, done)
            RLagent.learn()   
            observation = observation_
        scores.append(score)
        eps_history.append(RLagent.epsilon)

        avg_score = np.mean(scores)

        print('episode', j, 'score %.2f' %score,
				'average score %.2f' % avg_score,
				'epsilon %.2f' % RLagent.epsilon)
        
    
    T.save({
            'RLagent_epsilon': RLagent.epsilon,
            'model_state_dict': RLagent.Q_eval.state_dict(),
            'optimizer_state_dict': RLagent.Q_eval.optimizer.state_dict(),
            'RLagent_state_memory': RLagent.state_memory,
            'RLagent_new_state_memory': RLagent.new_state_memory,
            'RLagent_action_memory': RLagent.action_memory,
            'RLagent_reward_memory': RLagent.reward_memory,
            'RLagent_terminal_memory': RLagent.terminal_memory,
            'RLagent_mem_cntr': RLagent.mem_cntr
            }, FILE)
    
    

    
def plotRL():  
    scenario = dict(
        name='scenario',
        policy = 'RL',
    )
    
    RLagent = RLAgent(gamma=0.7, epsilon=1.0, batch_size=64, n_actions=5, eps_end=0.01, input_dims=[5], lr=0.003)
    FILE = "modelinfo.pth"
    checkpoint = T.load(FILE)
    RLagent.Q_eval.load_state_dict(checkpoint['model_state_dict'])
    RLagent.Q_eval.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    RLagent.epsilon = checkpoint['RLagent_epsilon']
    RLagent.state_memors = checkpoint['RLagent_state_memory']
    RLagent.new_state_memory = checkpoint['RLagent_new_state_memory']
    RLagent.action_memory = checkpoint['RLagent_action_memory']
    RLagent.reward_memory = checkpoint['RLagent_reward_memory']
    RLagent.terminal_memory = checkpoint['RLagent_terminal_memory']
    RLagent.mem_cntr = checkpoint['RLagent_mem_cntr']
    RLagent.epsilon = 0
    np.random.seed(1)
    sim = GraphSimulation(**{**scenario})
    from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                                   AutoMinorLocator)

    frames = 1440
    iteration_time = 25
    tick_unit = 72

    score = 0
    done = False
    observation = sim.observation 

    anim = execute_graphsimulationRL(sim, RLagent, iteration_time=iteration_time, iterations=frames)

    return(anim)
