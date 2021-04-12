import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_probability as tfp

from utils import Replay

"""
# Main wrapper function

* Function to run an algorithm on a specified environment
"""

def PlayRL(game_env_class, # way to progress through game
           perturb_game_env, # function to perturb the MDP (e.g. non-stationary MDPs)
           main_network_class, # main network is updated regularly (throughout the game) and used to select actions
           target_network_class, # target network is used as a proxy for actually knowing the q value
           ensemble_weight_class, # ensemble weights used to select over ensembles
           network_update, # loss needs to match what the networks look like
           weight_update, # how to update the weights
           play_name, # string to help store data cleanly
           q_sample = lambda x: x, # function to sample given parametrization of q vals, default identity for deterministic q vals
           num_episodes = 1000, # number of new games to play
           explore_action = 0.1, # exploration rate for the policy of a model
           explore_weight = 0.05, # exploration rate of models from the ensemble
           training_freq = 1, # episodes between training main network
           update_freq = 1, # episodes between updating target network
           weight_freq = 1, # episodes between updating ensemble weights
           net_lr = 1e-5, # learning rate for gradient optimization on networks
           weight_lr = 1e-5, # learning rate for gradient optimization on ensemble weights
           mem_length = 1000, # number of rounds to keep in memory for 
           mem_batch = 64, # number of rounds to sample during gradient update
           mem_weight = None, # number of rounds to keep for weights (if None keep all)
           discount = 0.99, # amount to discount future rewards
           seed = None, # seed for replication purposes
          ):
    
    # game environment
    game_env = game_env_class()
    num_actions = game_env.num_actions()
    optimal_episode_reward = game_env.optimal_reward()
    game_uniform = [1. / num_actions] * num_actions

    # set seed for replication purposes
    if seed is not None:
        np.random.seed(seed)
        tf.random.set_seed(seed)

    # agent objects
    main_network = main_network_class()
    target_network = target_network_class()
    num_ensemble = main_network.num_ensemble
    ensemble_weights = ensemble_weight_class(tf.Variable([1. / num_ensemble] * num_ensemble))
    agent_memory = Replay(capacity=mem_length)
    weight_history = []

    # optimizers
    net_optimizer = tf.optimizers.Adam(learning_rate=net_lr)
    weight_optimizer = tf.optimizers.Adam(learning_rate=weight_lr)

    # keep track of performance
    results = []
    active_nets = []
    episode_rewards = []
    weight_memory = []
    total_reward = 0
    optimal_reward = 0
    total_regret = 0

    print('\n**', 'Training phase: ', play_name, '**')
    for episode in range(num_episodes):

        # set up the episode
        timestep = game_env.reset()
        observation = timestep.observation.flatten()
        episode_reward = 0
        explore = np.random.binomial(n=1, p=explore_weight, size=1) # decide if forced exploration of ensemble
        if explore:
            active_net = tfp.distributions.Categorical(probs = tf.Variable([1. / num_ensemble] * num_ensemble)).sample()
        else:
            active_net = ensemble_weights.sample() # sample one of the ensemble nets to play with
        
        active_nets.append(active_net) # keep track of the choice
        
        # update target
        if episode % update_freq == 0:
            main_variables = main_network.trainable_variables
            target_variables = target_network.trainable_variables
            [v.assign(w.numpy()) for v, w in zip(target_variables, main_variables)]
        
        while timestep.continue_play:

            # agent takes action
            explore = np.random.binomial(n=1, p=explore_action, size=1) # decide if forced exploration

            if explore:
                action = np.random.multinomial(n=1, p=game_uniform, size=1) # uniformly select an action

            else:
                current_state = np.atleast_2d(np.atleast_2d(observation).astype('float32')) # convert observation to input state
                ensemble_qs = q_sample(main_network(current_state)) # receive prediction from each ensemble
                active_qs = ensemble_qs[:,active_net,:] # only select prediction from active net
                action = np.argmax(active_qs) # pick action with highest predicted q value

            # update environment
            timestep = game_env.step(action)
            episode_reward += timestep.reward
            new_observation = timestep.observation.flatten()

            # add to memory
            transition = (observation, action, timestep.reward, timestep.continue_play, new_observation)
            agent_memory.add(transition)
            observation = new_observation
        
        episode_rewards.append(episode_reward)
        weight_memory.append(ensemble_weights.probs_parameter().numpy())
        
        # update main network
        if episode % training_freq == 0:

            # sample a random batch from memory and train gradients
            batch_memory = agent_memory.sample(mem_batch)
            network_update(batch_memory, main_network, target_network, discount, net_optimizer, num_actions)

        # update ensemble weights
        if episode % weight_freq == 0 and episode > 1:
            
            # only keep certain amount of episode data for weights
            if mem_weight is not None:
                weight_memory = weight_memory[-min(len(weight_memory),mem_weight):]
            
            # use episode data to update weights
            ensemble_weights = weight_update(ensemble_weights, active_nets, episode_rewards, weight_memory, weight_optimizer)
        
        # perturb the game_env (to add non-stationarity)
        if perturb_game_env is not None:
            perturb_game_env(game_env)

        # update statistics
        total_reward += episode_reward
        optimal_reward += optimal_episode_reward
        total_regret = optimal_reward - total_reward

        results.append({'episode': episode, 'reward': total_reward, 'regret': total_regret, 'algorithm': play_name})
        weight_history.append(ensemble_weights)
        
        # print progress
        if episode % 2000 == 0 or episode == num_episodes - 1:
            print('Episode: {}.\tTotal reward: {:.2f}.\tRegret: {:.2f}.'.format(episode, total_reward, total_regret))
            print('Ensemble Weights: ', np.round(ensemble_weights.probs_parameter().numpy(), 2))
    
    # format weight table with pandas
    weight_memory = pd.DataFrame(weight_memory, columns=['w'+str(x) for x in range(0,num_ensemble)])
    weight_memory['episode'] = np.arange(len(weight_memory))
    weight_memory['algorithm'] = play_name
    
    return pd.DataFrame(results), main_network, weight_memory