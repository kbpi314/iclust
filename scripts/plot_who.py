import os
import numpy as np
import pandas as pd
import click
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import set_matplotlib_formats
sns.set_style("white")
set_matplotlib_formats('svg')


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

@click.command(context_settings=CONTEXT_SETTINGS)

@click.option('-i', '--input_df', required=True,
              type=str,
              help='Input txt file of coordinates')
@click.option('-r', '--r_matrix', required=True,
              type=str,
              help='Input r_matrix file of data')
@click.option('-o', '--output_dir', required=True,
              type=str,
              help='Output directory to store plots')

def plot_who(input_df, r_matrix, output_dir):
    '''
    '''
    # read in data
    # generating data from WHO
    r_matrix_df = pd.read_csv(r_matrix, sep='\t')
    r_matrix_df = r_matrix_df.loc[r_matrix_df['var1_index'] > r_matrix_df['var2_index']]

    small_df = r_matrix_df[(r_matrix_df['correlations'] > 0.1995) &
                           (r_matrix_df['correlations'] < 0.2005)]
    medium_df = r_matrix_df[(r_matrix_df['correlations'] > 0.499) &
                            (r_matrix_df['correlations'] < 0.501)] # for pearson
    large_df = r_matrix_df[(r_matrix_df['correlations'] > 0.8975) &
                           (r_matrix_df['correlations'] < 0.9025)]

    print(len(small_df))
    print(len(medium_df))
    print(len(large_df))
    df_list = [small_df, medium_df, large_df]

    dir_list = [os.path.join(output_dir, 'small_corr/'),
                os.path.join(output_dir, 'med_corr/'),
                os.path.join(output_dir, 'large_corr/')]

    plotting_data = pd.read_csv(input_df, sep='\t')

    plotting_data.columns = [str(x) for x in range(len(plotting_data.columns))]
    plotting_xnames = plotting_data.columns.values[3:]

    font = {'size'   : 2}
    matplotlib.rc('font', **font)
    for path in dir_list:
        try:
            os.makedirs(path)
        except:
            pass

    # for each type in 'dataset'
    #  plot them
    for i, current_df in enumerate(df_list):
        for index, row in current_df.iterrows():
            var1 = int(row['var1_index'])
            var2 = int(row['var2_index'])
            current_data = plotting_data[[plotting_xnames[var1],
                                          plotting_xnames[var2]]].dropna()

            # create figure
            fig = plt.figure()

            sns.lmplot(plotting_xnames[var1], plotting_xnames[var2],
                       data=current_data, fit_reg=False, palette="Set1")

            axes = plt.gca()
            axes.set_position(np.array([.1, .1, .8, .8]))
            plt.tick_params(axis='both', which='both', bottom=False, top=False,
                            left=False, right=False, labelbottom=False,
                            labelleft=False)
            axes.set_ylabel('')
            axes.set_xlabel('')
            axes.set_title('')
            plt.ticklabel_format(useOffset=False, style='plain', axis='both')
            fig.patch.set_visible(False)
            axes.patch.set_visible(False)
            plt.savefig(dir_list[i] + 'WHO_' + str(var1) + '_' + str(var2) + \
                '_' + str(round(float(row['correlations']), 2)) + '.jpg')

            plt.close(fig)

if __name__ == "__main__":
    plot_who()
