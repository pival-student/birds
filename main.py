from birdutils import *
# Import the libraries
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import math
import numpy as np

matplotlib.use('TkAgg')


def plot(data, key='', cut=3):
    print('Plotting..')
    lens_combined = sns.displot(data, x=key, kind='kde', hue='lang', rug=False, fill=False, cut=cut)
    lens_combined.savefig(f'plots/{key}_combined.png')
    plt.close(lens_combined.fig)
    # ax = lens_combined.axes.flatten()[0]
    for i, spec in enumerate(data['lang'].unique()):
        # add avg line to combined plot
        # ax.axvline(x=means.iloc[i], ymin=0, ymax=1, ls='-.', color=sns.color_palette()[i])
        # ax.text(means.iloc[i], 0.95, f'{round(means.iloc[i], 2)}', color=sns.color_palette()[i], ha='left', va='bottom',
        #         transform=ax.get_xaxis_transform())
        # create separate plots
        mean = data[data['lang'] == spec][key].mean()
        median = data[data['lang'] == spec][key].median()
        lens_individual = sns.displot(data[data['lang'] == spec], x=key, kind='hist', kde=True)
        axi = lens_individual.axes.flatten()[0]
        axi.axvline(x=mean, ymin=0, ymax=1, ls='-.')
        axi.text(mean, 0.95, f'm={round(mean, 2)}', ha='left', va='bottom',
                 transform=axi.get_xaxis_transform())
        axi.axvline(x=median, ymin=0, ymax=1, ls=':')
        axi.text(median, 0.85, f'md={round(median, 2)}', ha='right', va='bottom',
                 transform=axi.get_xaxis_transform())
        lens_individual.savefig(f'plots/{key}_{spec}.png')
        plt.close(lens_individual.fig)



if __name__ == '__main__':
    # get bird data
    robin = get_data_robin('data/robin/data')
    bhgr = get_data_singletier('data/birddb/bhgr')
    cath = get_data_singletier('data/birddb/cath2')
    cavi = get_data_singletier('data/birddb/cavi2012')
    # compute bird stats
    stats = {
        'robin': new_stats(robin, 'robin'),
        'bhgr': new_stats(bhgr, 'bhgr'),
        'cath': new_stats(cath, 'cath'),
        'cavi': new_stats(cavi, 'cavi')
    }
    # concatenate bird df
    bird_df = pd.concat([stats[lc]['data'] for lc in stats.keys()])
    # plot
    plot(bird_df, 'seq_len')
    plot(bird_df, 'seq_ttr', cut=0)
    plot(bird_df, 'seq_repr', cut=0)
    plot(bird_df, 'seq_entr_loc', cut=0)
    plot(bird_df, 'seq_entr_glob', cut=0)





    # stats = new_stats(robin, 'robin')
    # # print('Robin:')
    # res = sns.displot(stats['sent_reprs'], kind='kde', rug=True, fill=True)
    # # res = sns.displot(stats['sent_lens'], kind='hist', color='blue')
    #
    # plt.show()
    #
    # bg = get_data_singletier('data/birddb/cath2')
    #
    # stats = new_stats(bg, 'bhgr')
    # # print('Robin:')
    # res = sns.displot(stats['sent_reprs'], kind='kde', rug=True)
    # # res = sns.displot(stats['sent_lens'], kind='hist', color='blue')
    # res.savefig('plot-uncut.png')
    # # plt.show()

    # print(stats(robin))
    #
    # bghr = get_data_singletier('data/birddb/bhgr')
    # print('Black-headed Grosbeak:')
    # print(stats(bghr))
    #
    # cath = get_data_singletier('data/birddb/cath2')
    # print('California Thrasher:')
    # print(stats(cath))
    #
    # cavi = get_data_singletier('data/birddb/cavi2012')
    # print("Cassin's Vireo:")
    # print(stats(cavi))
    # data = get_data_human('data/teddi', r'deu.*\.txt')
    # tok_data = build_tokenized_data(data, False)
    # stats = compute_human_stats(tok_data)
    # print(stats)
