from birdutils import *
# Import the libraries
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import math
import numpy as np

matplotlib.use('TkAgg')

def print_stats(stats, prefix):
    for lc in stats.keys():
        with open(f'stats/{prefix}_{lc}.txt', 'w', encoding='utf-8') as inf:
            for key in ['seq-count', 'tokens', 'types', 'entropy', 'top-10', 'bot-10']:
                if key == 'top-10' or key == 'bot-10':
                    inf.write(f'{key}:\n')
                    tops = stats[lc][key]
                    for t in tops:
                        per = round(100 * t[1], 2)
                        inf.write(f'  "{t[0]}": {per}%\n')
                elif key != 'data':
                    inf.write(f'{key}: {stats[lc][key]}\n')
                else:
                    continue

    pass


def plot(data, task='', key='', cut=3):
    print('Plotting..')
    lens_combined = sns.displot(data, x=key, kind='kde', hue='lang', rug=False, fill=False, cut=cut)
    lens_combined.savefig(f'plots/{task}_{key}_combined.png')
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
    bird_stats = {
        'robin': new_stats(robin, 'robin'),
        'bhgr': new_stats(bhgr, 'bhgr'),
        'cath': new_stats(cath, 'cath'),
        'cavi': new_stats(cavi, 'cavi')
    }
    print_stats(bird_stats, 'bird')
    # concatenate bird df
    bird_df = pd.concat([bird_stats[lc]['data'] for lc in bird_stats.keys()])
    # plot
    plot(bird_df, 'birds', 'seq_len')
    plot(bird_df, 'birds', 'seq_ttr', cut=0)
    plot(bird_df, 'birds', 'seq_repr', cut=0)
    plot(bird_df, 'birds', 'seq_entr_loc', cut=0)
    plot(bird_df, 'birds', 'seq_entr_glob', cut=0)


    eng = get_data_human('data/teddi', r'eng.*\.txt')
    eng_docs, eng_lines = tokenize(eng['eng'], 'eng')
    eng_words = charsplit(eng_docs, 'eng')

    deu = get_data_human('data/teddi', r'deu.*\.txt')
    deu_docs, deu_lines = tokenize(deu['deu'], 'deu')
    deu_words = charsplit(deu_docs, 'deu')

    heb = get_data_human('data/teddi', r'heb.*\.txt')
    heb_docs, heb_lines = tokenize(heb['heb'], 'heb')
    heb_words = charsplit(heb_docs, 'heb')
    
    tur = get_data_human('data/teddi', r'tur.*\.txt')
    tur_docs, tur_lines = tokenize(tur['tur'], 'tur')
    tur_words = charsplit(tur_docs, 'tur')
    
    vie = get_data_human('data/teddi', r'vie.*\.txt')
    vie_docs, vie_lines = tokenize(vie['vie'], 'vie')
    vie_words = charsplit(vie_docs, 'vie')

    doc_stats = {
        'eng_docs': new_stats(eng_docs, 'eng_docs'),
        'deu_docs': new_stats(deu_docs, 'deu_docs'),
        'heb_docs': new_stats(heb_docs, 'heb_docs'),
        'tur_docs': new_stats(tur_docs, 'tur_docs'),
        'vie_docs': new_stats(vie_docs, 'vie_docs'),
    }
    print_stats(doc_stats, 'lang')
    line_stats = {
        'eng_lines': new_stats(eng_lines, 'eng_lines'),
        'deu_lines': new_stats(deu_lines, 'deu_lines'),
        'heb_lines': new_stats(heb_lines, 'heb_lines'),
        'tur_lines': new_stats(tur_lines, 'tur_lines'),
        'vie_lines': new_stats(vie_lines, 'vie_lines'),
    }
    print_stats(line_stats, 'lang')
    word_stats = {
        'eng_words': new_stats(eng_words[:1000000], 'eng_words'),
        'deu_words': new_stats(deu_words[:1000000], 'deu_words'),
        'heb_words': new_stats(heb_words[:1000000], 'heb_words'),
        'tur_words': new_stats(tur_words[:1000000], 'tur_words'),
        'vie_words': new_stats(vie_words[:1000000], 'vie_words'),
    }
    print_stats(word_stats, 'lang')

    doc_df = pd.concat([doc_stats[lc]['data'] for lc in doc_stats.keys()])
    # plot
    plot(doc_df, 'doc',  'seq_len')
    plot(doc_df, 'doc',  'seq_ttr', cut=0)
    plot(doc_df, 'doc',  'seq_repr', cut=0)
    plot(doc_df, 'doc',  'seq_entr_loc', cut=0)
    plot(doc_df, 'doc',  'seq_entr_glob', cut=0)

    line_df = pd.concat([line_stats[lc]['data'] for lc in line_stats.keys()])
    # plot
    plot(line_df, 'line',  'seq_len')
    plot(line_df, 'line',  'seq_ttr', cut=0)
    plot(line_df, 'line',  'seq_repr', cut=0)
    plot(line_df, 'line',  'seq_entr_loc', cut=0)
    plot(line_df, 'line',  'seq_entr_glob', cut=0)

    word_df = pd.concat([word_stats[lc]['data'] for lc in word_stats.keys()])
    # plot
    plot(word_df, 'word',  'seq_len')
    plot(word_df, 'word',  'seq_ttr', cut=0)
    plot(word_df, 'word',  'seq_repr', cut=0)
    plot(word_df, 'word',  'seq_entr_loc', cut=0)
    plot(word_df, 'word',  'seq_entr_glob', cut=0)


    
    








    # cs = get_data_charsets('data/teddi')
    # print(cs)

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
