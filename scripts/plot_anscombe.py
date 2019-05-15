import os, re
import numpy as np
import pandas as pd
import click
import scipy.stats
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('svg')
from collections import defaultdict
import glob

from scipy.spatial import distance
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import dendrogram, set_link_color_palette, linkage
from matplotlib.colors import rgb2hex, colorConverter
from sklearn.metrics import v_measure_score, silhouette_score
from PIL import Image
from iclust import imagecluster as ic
from iclust import common as co
from iclust import analysis

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

@click.command(context_settings=CONTEXT_SETTINGS)

@click.option('-i', '--input_df', required=True,
              type=str,
              help='Input txt file of coordinates')
@click.option('-o', '--output_dir', required=True,
              type=str,
              help='Output directory to store plots')

def plot_anscombe(input_df, output_dir):
    '''
    '''
    # set seed for reproducibility
    np.random.seed(0)

    # 10 simulations per class in the quartet
    n_sim = 10

    # arbitrary but reasonable specifications for noise
    mu = 0
    sigmas = [0, 0.1, 0.25, 0.5, 0.75, 1]

    # read in data
    df = pd.read_csv(input_df, sep = '\t')

    # for each noise level
    for sigma in sigmas:
        path = os.path.join(output_dir, 'aq' + str(n_sim) + 'x_' + str(sigma))
        if os.path.exists(path) is not True:
            os.makedirs(path)

        # key 1 is the group
        # key 2 is the simulation number
        # entry is the set of points (x,y) in an array
        sim_plots = dict()
        for i in df.group:
            sim_plots[i] = dict()
            rel_df = df[df['group'] == i].drop('group', axis = 1)
            for j in range(n_sim):
                sim_plots[i][j] = rel_df + np.random.normal(mu, sigma,
                    [len(rel_df),len(rel_df.columns)])

        # generate plots
        # for each type of plot in 'dataset'
        for class_name in df['group'].unique():
            # obtain relevant data
            current_set = sim_plots[class_name]
            for plot in current_set:
                # obtain relevant datas
                current_df = current_set[plot]
                pc, pp = scipy.stats.pearsonr(current_df['x'], current_df['y'])
                # create figure
                fig = plt.figure()
                ax = plt.gca()
                sns_plot = sns.lmplot('x', 'y',
                    data=current_df, fit_reg=False, palette="Set1")
                ax.set_position(np.array([.1,.1,.8,.8]))
                plt.tick_params(axis='both', which='both', bottom=False,
                    top=False, left=False, right=False, labelbottom=False,
                    labelleft=False)
                plt.ticklabel_format(useOffset=False, style='plain', axis='both')
                ax.set_ylabel('')
                ax.set_xlabel('')
                fig.patch.set_visible(False)
                ax.patch.set_visible(False)
                plt.savefig(os.path.join(path,
                    class_name + '_' + str(plot) + '_' + str(round(pc, 3)) + '.jpg'))
                plt.close(fig)

if __name__ == "__main__":
    plot_anscombe()

