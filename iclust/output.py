import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram
from sklearn.decomposition import PCA
from IPython.display import set_matplotlib_formats
sns.set_style("white")
set_matplotlib_formats('svg')

def plot_corr(var1, var2, data, fp, points, axis_on, fixaxis, axis_bounds=[]):
    '''
    Generates pairwise correlation plot
    '''
    fig = plt.figure()
    axes = plt.gca()
    if points:
        g = sns.lmplot(var1, var2, data=data, fit_reg=False, palette="Set1")
    else:
        g = sns.lineplot(var1, var2, data=data, palette="Set1")
    axes.set_position(np.array([.1, .1, .8, .8]))
    if fixaxis:
        g.set(xlim=(axis_bounds[0],axis_bounds[1]),
              ylim=(axis_bounds[2], axis_bounds[3]))
    plt.tick_params(axis='both', which='both', bottom=False,
                    top=False, left=False, right=False,
                    labelbottom=False, labelleft=False,
                    labelright=False, labeltop=False)
    plt.ticklabel_format(useOffset=False, style='plain', axis='both')
    axes.set_ylabel('')
    axes.set_xlabel('')
    if not axis_on:
        plt.axis('off')
    fig.patch.set_visible(False)
    axes.patch.set_visible(False)
    plt.savefig(fp)
    plt.close(fig)

def plot_dend(plot_label, linkages, ordered_imgs, output_dir):
    '''
    Plot dendrogram
    '''
    plt.subplots()
    plt.figure(figsize=(20, 10))
    plt.title('Hierarchical Clustering Dendrogram (' + plot_label + ')')
    plt.xlabel('plot name')
    plt.ylabel('distance (' + plot_label + ')')
    dend_results = dendrogram(linkages, labels=ordered_imgs, leaf_rotation=90)
    plt.savefig(os.path.join(output_dir, plot_label + '.pdf'))
    plt.close()
    return dend_results

def plot_pca(df, output_dir):
    '''
    Generate PCA
    '''
    X = df.drop('group',axis=1)
    y = df['group']
    pca = PCA(n_components=3)
    pca.fit(X)
    ev = pca.explained_variance_ratio_
    pca = PCA(2)
    projected = pca.fit_transform(X)
    colors = []
    unique_vals = df['group'].unique()
    random.seed(a=0)
    for i in range(len(unique_vals)):
        colors.append('#%06X' % random.randint(0, 0xFFFFFF))

    points = {}
    for v in range(len(unique_vals)):
        points[v] = plt.scatter(projected[:, 0][df['group'] == unique_vals[v]],
                                projected[:, 1][df['group'] == unique_vals[v]],
                                #projected[:, 2],
                                c=colors[v],
                                edgecolor='none', alpha=0.5)  #cmap=plt.cm.get_cmap('Spectral', 10))\n",
        plt.xlabel('component 1')
        plt.ylabel('component 2')
        plt.title('PC explained var: ' + str(ev),  fontsize=8)
        plt.legend([points[v] for v in points], unique_vals)
    plt.savefig(os.path.join(output_dir, 'iclust_pca.pdf'))
    plt.close('all')

def print_results(best_ss_df, best_vms_df, output_dir):
    '''
    Output .txt file with clustering results
    '''
    df = pd.DataFrame(columns=['nss', 'ss', 'nvms', 'vms'], dtype=np.float64)
    df.loc[0] = np.array([
        best_ss_df['n_clust'].values[0],
        best_ss_df['ss'].values[0],
        best_vms_df['n_clust'].values[0],
        best_vms_df['vms'].values[0]])
    df.to_csv(output_dir + 'results_df.txt', sep='\t')

    best_k = best_vms_df['n_clust'].values[0]
    return best_k
