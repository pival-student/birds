from birdutils import *
# Import the libraries
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
matplotlib.use('TkAgg')

if __name__ == '__main__':
    robin = get_data_robin('data/robin/data')

    stats = new_stats(robin, 'robin')
    # print('Robin:')
    res = sns.displot(stats['sent_reprs'], kind='kde', rug=True)
    # res = sns.displot(stats['sent_lens'], kind='hist', color='blue')

    plt.show()

    bg = get_data_singletier('data/birddb/cath2')

    stats = new_stats(bg, 'bhgr')
    # print('Robin:')
    res = sns.displot(stats['sent_reprs'], kind='kde', rug=True)
    # res = sns.displot(stats['sent_lens'], kind='hist', color='blue')

    plt.show()


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

