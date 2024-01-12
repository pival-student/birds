from utils import *

if __name__ == '__main__':
    # robin = get_data_robin('data/robin/data')
    # print('Robin:')
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
    data = get_data_human('data/teddi', r'deu.*\.txt')
    tok_data = build_tokenized_data(data, False)
    stats = compute_human_stats(tok_data)
    print(stats)

