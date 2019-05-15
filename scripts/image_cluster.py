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

def image_cluster(input_dir, output_dir, labeled, max_clust):
    """Example main app using this library.

    Parameters
    ----------
    input_dir : str
        path to directory with images
    sim : float (0..1)
        similarity index (see imagecluster.cluster())
    """
    dbfn = os.path.join(output_dir, 'data_analysis/fingerprints.pk')
    if not os.path.exists(dbfn):
        os.makedirs(os.path.dirname(dbfn), exist_ok=True)
        print("no fingerprints database {} found".format(dbfn))

        # obtain files
        files = co.get_files(input_dir)

        # retrieve transfer learning model
        model = ic.get_model()
        print("running all images through NN model ...".format(dbfn))

        # fps is a dictionary mapping image to model fingerprint
        # files is a list of full paths, but the keys of fps are basenames
        fps = ic.fingerprints(files, model, size=(224,224))
        co.write_pk(fps, dbfn)
    else:
        print("loading fingerprints database {} ...".format(dbfn))
        fps = co.read_pk(dbfn)

    print("clustering ...")
    # perform distance calculations
    # dfps is a condensed distance matrix
    # Z is a linkage matrix

    dfps = distance.pdist(np.array(list(fps.values())), metric='euclidean')
    # ordered_imgs is a list of file names as strings corresponding to the distance matrix axis
    ordered_imgs = list(fps.keys())
    # hierarchical/agglomerative clustering (Z = linkage matrix, construct
    # dendrogram)
    Z = hierarchy.linkage(dfps, method='average', metric='euclidian')

    # take condensed distance matrix dfps and make it square e.g. noncondensed
    noncond_dist = distance.squareform(dfps)

    # determine best clustering
    ss, vms, best_ss_df, best_vms_df, corr_score_df = analysis.score_clusters(
        ordered_imgs, Z, noncond_dist, max_clust, labeled)

    print('best ss n_clust: ' + str(best_ss_df['n_clust'].values[0]) + ' with score ' +  str(best_ss_df['ss'].values[0]))
    print('best vms n_clust: ' + str(best_vms_df['n_clust'].values[0]) + ' with score ' +  str(best_vms_df['vms'].values[0]))
    best_k = best_vms_df['n_clust'].values[0] # toggle here

    # use image clustering
    # clustered_imgs is a flattened list of the nested list of images in clusters corresponding to the cut made
    # cluster_to_img is a dict mapping cluster number to list of images
    # img_to_cluster is a dict mapping image to cluster number
    clustered_imgs, cluster_to_img, img_to_cluster = ic.cluster(ordered_imgs, Z, best_k)

    # assigned labels corresponding to flattened list
    flat_list = [item for sublist in clustered_imgs for item in sublist]
    assigned_labels = [img_to_cluster[x] for x in flat_list]

    # create dendrograms
    fig, ax = plt.subplots()
    plt.figure(figsize=(20,10))
    plt.title('Hierarchical Clustering Dendrogram (imageclust)')
    plt.xlabel('plot name')
    plt.ylabel('distance (imageclust)')
    R = dendrogram(Z, labels = ordered_imgs, leaf_rotation = 90)
    plt.savefig(os.path.join(output_dir, 'data_analysis/iclust.png'))
    plt.close()

    analysis.print_avg_order(ordered_imgs, Z, best_k, input_dir, os.path.join(output_dir, 'data_analysis/'), 'iclust order', R)
    # R['ivl'] = [list(x) for x in R['ivl']]
    ic.make_links(clustered_imgs, os.path.join(input_dir, 'data_analysis/clusters'))

if __name__ == "__main__":
    image_cluster()

