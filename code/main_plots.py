import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import plotnine as gg

from utils import directory

# important variables from main.py
num_ensemble = 10
num_sims = 1
repo_import = 'csv_tables'

# create a repository for plots
repo_export = 'plots'
directory(repo_export)

for environment in ["DeepSea","CartPole"]: # ["DeepSea","CartPole"]
    for nonstationary in ["", "Perturbed_"]: # ["", "Perturbed_"]
        # import result tables (after running main.py with the desired parameters)
        performance = pd.read_csv(repo_import + '/performance' + nonstationary + environment + '_' + str(num_sims) + 'sims.csv')
        weights = pd.read_csv(repo_import + '/weights' + nonstationary + environment + '_' + str(num_sims) + 'sims.csv')
        
        performance['algorithm'] = performance['algorithm'].str.replace(' ' + nonstationary + environment, '')
        weights = pd.wide_to_long(weights, ["w"], i=['algorithm','simulation','episode'], j='model').reset_index()
        weights['model'] = weights['model'].astype('object')

        # obtain the id of one simulation
        sim = int(np.random.randint(0, np.max(performance['simulation'])+1, size=1))
        
        # plot cumulative regret for a single simulation
        regret_plot = (gg.ggplot(performance[performance['simulation']==sim])
                    + gg.aes(x='episode', y='regret', fill='algorithm', colour='algorithm')
                    + gg.geom_line(size=1.3, alpha=0.7)
                    + gg.xlab('Episodes')
                    + gg.ylab('Cumulative regret')
                    )
        regret_plot.save(filename = repo_export + '/regret_' + nonstationary + environment + '_'+ str(num_sims) + 'sims.pdf', 
                         height=5, width=5, units = 'in', dpi=2000)
        
        if num_sims > 1:
            # plot confidence intervals with multiple simulations
            regret_plot_CI = (gg.ggplot(performance)
                        + gg.aes(x='episode', y='regret', fill='algorithm', colour='algorithm')
                        + gg.stat_summary(fun_data='mean_sdl', fun_args={"mult": 1}, alpha=0.01)
                        + gg.xlab('Episodes')
                        + gg.ylab('Cumulative regret')
                        + gg.theme(legend_position='none')
                        )
            regret_plot_CI.save(filename = repo_export + '/regret_CI_' + nonstationary + environment + '_'+ str(num_sims) + '.pdf', 
                             height=5, width=5, units = 'in', dpi=2000)
        
        # evolution of weights
        for weight_loss in ["Unif", "PG", "Exp3"]:
            weight_evolution = (gg.ggplot(weights[(weights["simulation"]==sim) & \
                                                  (weights["algorithm"]==(str(num_ensemble) + '-NeuralNets ' + nonstationary + environment + ' ' + weight_loss + '-weights'))])
                        + gg.aes(x='episode', y='w', fill='model', colour='model')
                        + gg.geom_line(size=1.3, alpha=0.7)
                        + gg.xlab('Episodes')
                        + gg.ylab('Weight')
                        )
            weight_evolution.save(filename = repo_export + '/weight_evol_' + nonstationary + environment + '_'+ str(num_sims) + 'sims_' + weight_loss + '.pdf', 
                             height=5, width=5, units = 'in', dpi=2000)