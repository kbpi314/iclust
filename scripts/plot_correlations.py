import os
import numpy as np
import pandas as pd
import click
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
from IPython.display import set_matplotlib_formats
from iclust import output
sns.set_style("white")
set_matplotlib_formats('svg')


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

@click.command(context_settings=CONTEXT_SETTINGS)

@click.option('-i', '--input_df', required=True,
              type=str, help='Input txt file of coordinates')
@click.option('-o', '--output_dir', required=True,
              type=str, help='Output directory to store plots')
@click.option('--labeled/--unlabeled', required=True, default=False,
              help='Boolean for whether data is labeled or unlabeled')
@click.option('-l', '--true_label', required=True, default='group',
              type=str, help='String denoting true class')
@click.option('-lb', '--lower_bound', required=False, default=-1, type=float,
              help='Float denoting lower bound of correlations to plot')
@click.option('-ub', '--upper_bound', required=False, default=1, type=float,
              help='Float denoting upper bound of correlations to plot')
@click.option('-s', '--n_seed', required=False,
              type=int, default=0,
              help='Seed for random number generator')
@click.option('-n', '--n_sim', required=False,
              type=int, default=10,
              help='Number of simulations')
@click.option('-sv', '--sigma_vector', required=False,
              type=str, default='0.1,0.5,1',
              help='CSV string with values for noise')
@click.option('-dl', '--data_label', required=False,
              type=str, help='String value for label of dataset')

def plot_correlations(input_df, output_dir, labeled, true_label, lower_bound,
                      upper_bound, n_seed, n_sim, sigma_vector, data_label):
    '''
    '''
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # read in data
    if labeled:
        # if data is labeled, should have format header ['x', 'y', <true_label>] in any order
        np.random.seed(n_seed)

        # noise specification
        mean = 0
        sigmas = [float(x) for x in sigma_vector.split(',')]

        # read in data
        simulated_df = pd.read_csv(input_df, sep='\t')

        # can't have underscores
        simulated_df[true_label] = simulated_df[true_label].str.replace('_', "-")

        for sigma in sigmas:
            path = os.path.join(output_dir, data_label + '_' + str(n_seed) + '_' + str(n_sim) + 'x_' + str(sigma))
            if not os.path.exists(path):
                os.makedirs(path)

            # key 1 is the group
            # key 2 is the simulation number
            # entry is the set of points (x,y) in an array
            sim_plots = dict()
            for i in simulated_df[true_label]:
                sim_plots[i] = dict()
                rel_df = simulated_df[simulated_df[true_label] == i].drop(true_label, axis=1)
                for j in range(n_sim):
                    sim_plots[i][j] = rel_df + np.random.normal(mean, sigma,
                                                                [len(rel_df),
                                                                 len(rel_df.columns)])
            # generate plots
            # for each type of plot in 'dataset'
            for class_name in simulated_df[true_label].unique():
                # obtain relevant data
                current_set = sim_plots[class_name]
                for plot in current_set:
                    # obtain relevant datas
                    current_df = current_set[plot]
                    corr = scipy.stats.pearsonr(current_df['x'], current_df['y'])[0]

                    # create figure
                    fp = os.path.join(path, class_name + '_' + str(plot) + \
                                      '_' + str(round(corr, 3)) + '.jpg')
                    output.plot_corr('x', 'y', current_df, fp)


    else:
        # if data is unlabeled
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # input dataframe should be in format where rows are samples and columns are var
        df = pd.read_csv(input_df, sep='\t', index_col=0)
        df.columns = [str(x) for x in range(len(df.columns))]

        var_names = df.columns.values
        n_var = len(var_names)
        corrs = np.zeros([n_var, n_var])
        pairs = []
        for i in range(n_var):
            for j in range(i):
                current_data = df[[var_names[i], var_names[j]]].dropna()
                try:
                    corrs[i][j] = scipy.stats.pearsonr(current_data[var_names[i]],
                                                       current_data[var_names[j]])[0]
                except ValueError:
                    corrs[i][j] = 0
                if corrs[i][j] >= lower_bound and corrs[i][j] <= upper_bound:
                    pairs.append((i,j))

        print(str(len(pairs)) + ' plots were plotted')
        # font = {'size'   : 2}
        # matplotlib.rc('font', **font)

        # for each type in 'dataset' plot them
        for pair in pairs:
            var1, var2 = pair
            current_data = df[[var_names[var1], var_names[var2]]].dropna()
            fp = output_dir + data_label + '_' + str(var1) + '_' + str(var2) + \
                '_' + str(round(float(corrs[var1][var2]), 2)) + '.jpg'

            output.plot_corr(var_names[var1], var_names[var2], current_data, fp)


if __name__ == "__main__":
    plot_correlations()