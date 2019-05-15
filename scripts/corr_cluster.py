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

@click.option('-i', '--input_dir', required=True,
              type=click.Path(exists=True),
              help='Input  directory of images')
@click.option('-o', '--output_dir', required=True,
              type=click.Path(exists=True),
              help='Output  directory to store files during analysis')
@click.option('-l', '--labeled', required=True,
              type=bool,
              help='True if input images have true labels (format: <label>_blah.jpg)')
@click.option('-m', '--max_clust', required=True,
              type=int,
              help='Maximum number of clusters to iterate over when determining best')

def corr_cluster(input_dir, output_dir, labeled, max_clust):
    """Example main app using this library.

    Parameters
    ----------
    input_dir : str
        path to directory with images
    sim : float (0..1)
        similarity index (see imagecluster.cluster())
    """

    ordered_imgs = co.get_files(input_dir)

    # filename, file_extension = os.path.splitext
    # example basename: B_6_0.802.jpg
    corrs = [float(os.path.splitext(os.path.basename(x))[0].split('_')[-1]) for x in ordered_imgs]

    # 0 is used as a placeholder to get 2D array as input
    X = np.transpose(np.stack((np.array(corrs), np.zeros(len(corrs)))))

    # obtain pairwise distances
    condensed_dist = distance.pdist(X, metric='euclidean')

    # Calculate the distance between each sample, input must be condensed matrix
    Z = linkage(condensed_dist, 'ward')

    # convert from condensed to 2D for silhouette score
    noncond_dist = distance.squareform(condensed_dist)

    # determine best clustering
    ss, vms, best_ss_df, best_vms_df = analysis.score_clusters(
        ordered_imgs, Z, noncond_dist, max_clust, labeled)

    # output results
    print('best ss n_clust: ' + str(best_ss_df['n_clust'].values[0]) + ' with score ' +  str(best_ss_df['ss'].values[0]))
    print('best vms n_clust: ' + str(best_vms_df['n_clust'].values[0]) + ' with score ' +  str(best_vms_df['vms'].values[0]))
    best_k = best_vms_df['n_clust'].values[0] # toggle here

    # create dendrograms
    fig, ax = plt.subplots()
    plt.figure(figsize=(20,10))
    plt.title('Hierarchical Clustering Dendrogram (corr clust)')
    plt.xlabel('plot name')
    plt.ylabel('distance (corr clust)')
    R = dendrogram(Z, labels = ordered_imgs, leaf_rotation = 90)
    plt.savefig(os.path.join(output_dir, 'corrclust.png'))
    plt.close()

    analysis.print_avg_order(ordered_imgs, Z, best_k, input_dir, output_dir, 'corrclust order', R)

if __name__ == "__main__":
    corr_cluster()

