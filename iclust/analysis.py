import pandas as pd
import numpy as np
import os
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

# from imagecluster import main

# functions for image processing and output
def avg_image(output_dir, imlist, string):
    w,h=Image.open(imlist[0]).size
    N=len(imlist)

    # Create a numpy array of floats to store the average (assume RGB images)
    arr=np.zeros((h,w,3),np.float)

    # Build up average pixel intensities, casting each image as an array of floats
    for im in imlist:
        imarr=np.array(Image.open(im),dtype=np.float)
        arr=arr+imarr/N

    # Round values in array and cast as 8-bit integer
    arr=np.array(np.round(arr),dtype=np.uint8)

    # Generate, save and preview final image
    out=Image.fromarray(arr,mode="RGB")
    out.save(os.path.join(output_dir, string) + '.jpg')

def corr_order(input_dir, output_dir, R, string):
    #try:
    image_list = [os.path.join(input_dir, x) for x in R['ivl']]
    #except:
    #   new_R = [str(x[0]) for x in R['ivl']]
    #   image_list = [path + 'plots/' + x for x in new_R]

    images = list(map(Image.open, image_list))
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]

    new_im.save(os.path.join(output_dir, string) + '.jpg')

def score_clusters(ordered_imgs, Z, noncond_dist, max_clust, labeled):
    # groups is the truth, labels is the predicted
    # Z is linkage matrix
    # ids is the names of the files corresponding to the order in which Z was created
    ss = []
    vms = []
    n_clust = []

    for k in range(2, max_clust+1):
        cut = hierarchy.fcluster(Z, k, criterion='maxclust')
        cluster_to_img = defaultdict(list)
        img_to_cluster = {}
        for iimg,iclus in enumerate(cut):
            cluster_to_img[iclus].append(ordered_imgs[iimg])
            img_to_cluster[ordered_imgs[iimg]] = iclus

        clustered_imgs = list(cluster_to_img.values())
        # flatten list and extract labels
        flat_list = [item for sublist in clustered_imgs for item in sublist]
        assigned_labels = [img_to_cluster[x] for x in flat_list]

        # get assigned labels
        # example 'path/A_3_0.691.jpg'
        # A is the label, 3 is the replicate, 0.691 is the empirical corr
        true_labels = [x.split('_')[0] for x in flat_list]

        # if len(np.unique(assigned_labels)) > 1:
        ss.append(silhouette_score(noncond_dist, assigned_labels, metric='euclidean'))
        if labeled:
            vms.append(v_measure_score(true_labels, assigned_labels))
        else:
            vms.append(-1)

        n_clust.append(len(clustered_imgs))
        #else:
        #    ss.append(-1)
        #    vms.append(-1)
        #    n_clust.append(1)

    corr_score_df = pd.DataFrame({'ss': ss, 'vms': vms, 'n_clust': n_clust})
    best_ss_df = corr_score_df.loc[corr_score_df['ss'] == max(corr_score_df['ss'])]
    best_vms_df = corr_score_df.loc[corr_score_df['vms'] == max(corr_score_df['vms'])]
    return ss, vms, best_ss_df, best_vms_df

def print_avg_order(ordered_imgs, Z, k, input_dir, output_dir, string, R):
    cut = hierarchy.fcluster(Z, k, criterion='maxclust')
    cluster_to_img = defaultdict(list)
    img_to_cluster = {}
    for iimg,iclus in enumerate(cut):
        cluster_to_img[iclus].append(ordered_imgs[iimg])
        img_to_cluster[ordered_imgs[iimg]] = iclus

    clustered_imgs = list(cluster_to_img.values())

    i = 0
    for cluster in clustered_imgs:
        image_list = [os.path.join(input_dir, x) for x in cluster]
        i += 1
        avg_image(output_dir, image_list, 'avg_img_' + str(k) + '_' + str(i))

    corr_order(input_dir, output_dir, R, string)

