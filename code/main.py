import numpy as np
import pandas as pd
import functools

from envs import * # DeepSea, CartPole, Perturb_DeepSea, Perturb_CartPole
from play_wrapper import PlayRL
from models import * # EnsembleQNetwork, ModelWithPrior, EnsembleWithPrior, NoisyDense, EnsembleQNoisyNet, GaussianParamNet_2D, Ensemble_GaussianParamNet_2D
from losses import * # EnsembleWithPrior_Loss, Ensemble_GaussianParamNet_2D_Loss, Ensemble_GaussianParamNet_2D_Sample, Weight_Unif_Loss, Weight_PG_Loss, Weight_Exp3_Loss
from utils import directory 

# hyperparameters for the learning procedure
num_episodes = 10000 #@param {type:"integer"}
explore_action = 0 #@param {type:"number"}
explore_weight = 0 #@param {type:"number"}
training_freq = 1 #@param {type:"integer"}
update_freq = 1 #@param {type:"integer"}
weight_freq = 100 #@param {type:"integer"}
net_lr = 0.001 #@param {type:"number"}
weight_lr = 0.05 #@param {type:"number"}
mem_length = 100000 #@param {type:"integer"}
mem_batch = 64 #@param {type:"integer"}
mem_weight = None #@param {type:"raw"}
discount = 0.99 #@param {type:"number"}

# hyperparameters of all environments
game_size = 20 #@param {type:"integer"}
noisy_prob = 0.1 #@param {type:"number"}
gravity = 9.8 #@param {type:"number"}
length = 0.5 #@param {type:"number"}
gravity_drift = 0.95 #@param {type:"number"}
gravity_vol = 0.3 #@param {type:"number"}
length_drift = 0.99 #@param {type:"number"}
length_vol = 0.1 #@param {type:"number"}

# hyperparameters for the simulation study
num_ensemble = 10 #@param {type:"integer"}
num_sims = 1 #@param {type:"integer"}
seed = 4273 if num_sims==1 else None


# initialize game environments
game_env_DS = functools.partial(DeepSea, size=game_size, prob=0.5)
game_env_CP = functools.partial(CartPole, gravity=gravity,
                                          half_length=length)
perturb_DS = functools.partial(Perturb_DeepSea, noisy_prob=noisy_prob)
perturb_CP = functools.partial(Perturb_CartPole, gravity_drift =gravity_drift,
                                                 gravity_vol = gravity_vol,
                                                 length_drift = length_drift,
                                                 length_vol = length_vol)

# create a repository for results
repo = 'csv_tables'
directory(repo)

print('\n** Start of training phase **\n')

# experiments on all environments
for environment in ["DeepSea","CartPole"]: #["DeepSea","CartPole"]
    
    # assign specific values for each case
    if environment == "DeepSea":
        env_class = game_env_DS
        perturb_fun = perturb_DS
        input_size = game_size**2
    elif environment == "CartPole":
        env_class = game_env_CP
        perturb_fun = perturb_CP
        input_size = 6
    
    # experiments on stationary and non-stationary versions
    for nonstationary in ["", "Perturbed_"]: #["", "Perturbed_"]
        
        # assign specific values for each case
        if nonstationary == "":
            nonstationary_fun = None
        elif nonstationary == "Perturbed_":
            nonstationary_fun = perturb_fun
        
        # initialize table of results
        performance = pd.DataFrame([])
        weights = pd.DataFrame([])
        
        # loop for all simulations in order to construct confidence bands
        for sim in range(num_sims):
            print('Simulation: ', sim, '\n')
            
            # experiments on all networks
            for model in ["NeuralNet", "NoisyNet", "GaussianNet"]: #["NeuralNet", "NoisyNet", "GaussianNet"]
                
                # assign specific values for each case
                if model == "NeuralNet":
                    model_class = functools.partial(EnsembleWithPrior, prior_scale=10)
                    model_loss = EnsembleWithPrior_Loss
                    q_sample = lambda x: x
                elif model == "NoisyNet":
                    model_class = EnsembleQNoisyNet
                    model_loss = EnsembleWithPrior_Loss
                    q_sample = lambda x: x
                elif model == "GaussianNet":
                    model_class = Ensemble_GaussianParamNet_2D
                    model_loss = Ensemble_GaussianParamNet_2D_Loss
                    q_sample = Ensemble_GaussianParamNet_2D_Sample
                
                # Single network
                performance_history, _, weights_history =\
                    PlayRL(game_env_class = env_class,
                            perturb_game_env = nonstationary_fun,
                            main_network_class = functools.partial(model_class, input_size=input_size, hidden_sizes=[20,20], output_size=2, num_ensemble=1), 
                            target_network_class = functools.partial(model_class, input_size=input_size, hidden_sizes=[20,20], output_size=2, num_ensemble=1),
                            q_sample = q_sample,
                            ensemble_weight_class = tfp.distributions.Categorical,
                            network_update = model_loss,
                            weight_update = Weight_Unif_Loss,
                            play_name = '1-' + model + ' ' + nonstationary + environment,
                            num_episodes = num_episodes,
                            explore_action = explore_action,
                            explore_weight = explore_weight,
                            net_lr = net_lr,
                            weight_lr = weight_lr,
                            weight_freq = weight_freq,
                            mem_length = mem_length,
                            mem_batch = mem_batch,
                            mem_weight = mem_weight,
                            discount = discount,
                            seed = seed)
                performance_history["simulation"] = sim
                performance = performance.append(performance_history)
                
                for weight_loss in ["Unif", "PG", "Exp3"]: # ["Unif", "PG", "Exp3"]
                    
                    # assign specific values for each case
                    if weight_loss == "Unif":
                        weight_loss_function = Weight_Unif_Loss
                    elif weight_loss == "PG":
                        weight_loss_function = Weight_PG_Loss
                    elif weight_loss == "Exp3":
                        weight_loss_function = functools.partial(Weight_Exp3_Loss, exp3_lr = 0.005, exp3_eps=0)
                    
                    # Single network
                    performance_history, _, weights_history =\
                        PlayRL(game_env_class = env_class,
                                perturb_game_env = nonstationary_fun,
                                main_network_class = functools.partial(model_class, input_size=input_size, hidden_sizes=[20,20], output_size=2, num_ensemble=num_ensemble), 
                                target_network_class = functools.partial(model_class, input_size=input_size, hidden_sizes=[20,20], output_size=2, num_ensemble=num_ensemble),
                                q_sample = q_sample,
                                ensemble_weight_class = tfp.distributions.Categorical,
                                network_update = model_loss,
                                weight_update = weight_loss_function,
                                play_name = str(num_ensemble) + '-' + model + 's ' + nonstationary + environment + ' ' + weight_loss + '-weights',
                                num_episodes = num_episodes,
                                explore_action = explore_action,
                                explore_weight = explore_weight,
                                net_lr = net_lr,
                                weight_lr = weight_lr,
                                weight_freq = weight_freq,
                                mem_length = mem_length,
                                mem_batch = mem_batch,
                                discount = discount,
                                seed = seed)
                    performance_history["simulation"] = sim
                    performance = performance.append(performance_history)
                    weights_history["simulation"] = sim
                    weights = weights.append(weights_history)
        
        # store tables in .csv format
        performance.to_csv(repo + '/performance' + nonstationary + environment + '_'+ str(num_sims) + 'sims.csv', sep=',')
        weights.to_csv(repo + '/weights' + nonstationary + environment + '_' + str(num_sims) + 'sims.csv', sep=',')

print('\n** End of training phase **\n')