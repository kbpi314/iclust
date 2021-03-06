import matplotlib
matplotlib.use('Agg')
import os
import numpy as np
import click
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import set_matplotlib_formats
import random

from scipy.spatial import distance
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import linkage
from iclust import imagecluster as ic
from iclust import common as co
from iclust import output
from iclust import analysis

sns.set_style("white")
set_matplotlib_formats('svg')

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
@click.option('-min', '--min_clust', required=True,
              type=int,
              help='Minimum number of clusters to iterate over when determining best')
@click.option('-max', '--max_clust', required=True,
              type=int,
              help='Maximum number of clusters to iterate over when determining best')
@click.option('-m', '--method', required=True,
              type=str,
              help='String specifying type of analysis, e.g. nn for neural net, cc for corr str, rmse etc')

def image_cluster(input_dir, output_dir, labeled, min_clust, max_clust, method):
    """
    Parameters
    ----------
    input_dir : str
        path to directory with images
    output_dir : str
        path to directory for output files
    labeled : boolean
        TRUE if data have true known labels, false otherwise
    min_clust : int
        Minimum number of cluster numbers to iterate over
        e.g. range(min_clust. max_clust+1)
    max_clust : int
        Maximum number of cluster numbers to iterate over
    method : str
        String describing analysis (nn, cc, rmse)
    """

    if method == 'nn':
        plot_label = 'iclust'
        dbfn = os.path.join(output_dir, 'fingerprints.pk')
        if not os.path.exists(dbfn):
            os.makedirs(os.path.dirname(dbfn), exist_ok=True)
            print("no fingerprints database {} found".format(dbfn))

            # obtain files
            files = co.get_files(input_dir)

            print("running all images through NN model ...")
            # retrieve transfer learning model
            # fps is a dictionary mapping image to model fingerprint
            # files is a list of full paths, but the keys of fps are basenames
            fps = ic.fingerprints(files, model=ic.get_model(), size=(224, 224))
            co.write_pk(fps, dbfn)
        else:
            print("loading fingerprints database {} ...".format(dbfn))
            fps = co.read_pk(dbfn)

        print("clustering ...")
        # perform distance calculations
        # dfps is a condensed distance matrix
        # Z is a linkage matrix

        dfps = distance.pdist(np.array(list(fps.values())), metric='euclidean')

        # ordered_imgs is a list of file basenames as strings corresponding to the
        # distance matrix axis
        ordered_imgs = list(fps.keys())
        # hierarchical/agglomerative clustering (Z = linkage matrix, construct
        # dendrogram)

        linkages = hierarchy.linkage(dfps, method='average', metric='euclidian')

        # take condensed distance matrix dfps and make it square e.g. noncondensed
        noncond_dist = distance.squareform(dfps)

    elif method == 'cc':
        plot_label = 'cclust'
        ordered_imgs = co.get_files(input_dir)

        # filename, file_extension = os.path.splitext
        # example basename: B_6_0.802.jpg
        corrs = [float(os.path.splitext(os.path.basename(x))[0].split('_')[-1]) \
                 for x in ordered_imgs]

        # 0 is used as a placeholder to get 2D array as input
        data = np.transpose(np.stack((np.array(corrs), np.zeros(len(corrs)))))

        # obtain pairwise distances
        condensed_dist = distance.pdist(data, metric='euclidean')

        # Calculate the distance between each sample, input must be condensed matrix
        linkages = linkage(condensed_dist, 'ward')

        # convert from condensed to 2D for silhouette score
        noncond_dist = distance.squareform(condensed_dist)


    elif method == 'rmse':
        # replace plot_label with method string as is redundant
        plot_label = 'rmse'
        ordered_imgs = co.get_files(input_dir)

        # get an array of rmse in the same order of the ordered_imgs



        # subsequent 3 lines may be condensed
        # obtain pairwise distances between RMSEs
        condensed_dist = distance.pdist(data, metric='euclidean')

        # Calculate the distance between each sample, input must be condensed matrix
        linkages = linkage(condensed_dist, 'ward')

        # convert from condensed to 2D for silhouette score
        noncond_dist = distance.squareform(condensed_dist)

    # determine best clustering
    best_ss_df, best_vms_df = analysis.score_clusters(
        ordered_imgs, linkages, noncond_dist, min_clust, max_clust, labeled)

    best_k = output.print_results(best_ss_df, best_vms_df, output_dir)

    # create dendrograms
    dend_results = output.plot_dend(plot_label, linkages, ordered_imgs, output_dir)
    # print(dend_results['ivl'])

    if method == 'nn':
        # create pca
        # construct a df from the fingerprints and have a group label corresponding to original class
        indices = []
        values = []
        for img, fp in fps.items():
            indices.append(img)
            values.append(fp)

        img_to_cluster = analysis.print_avg_order(ordered_imgs, linkages,
            best_k, input_dir, output_dir, 'iclust order', dend_results)
        # print(img_to_cluster)

        df = pd.DataFrame(data=np.stack(values, axis=0), index=indices)
        if labeled:
            df['group'] = [str(x).split('_')[0] for x in df.index.values]
        else:
            df['group'] =['cluster ' + str(img_to_cluster[str(x)]) for x in df.index.values]

        output.plot_pca(df, output_dir)

if __name__ == "__main__":
    image_cluster()
