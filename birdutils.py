import re
import os
import math
import textgrids
from statistics import mean, median
from collections import Counter
from tqdm.auto import tqdm


def new_stats(list_of_tokenlists, lc):
    all = Counter()
    ttrs = []
    lengths = []
    reprs = []
    entrs = []
    entrs2 = []
    for sentence in tqdm(list_of_tokenlists, desc=f'{lc}: collecting counts'):
        types = Counter(sentence)
        lengths.append(len(sentence))
        all.update(sentence)
        # for token in sentence:
        #     types.update(token)
        ttrs.append(round(1.0 * len(types) / len(sentence), 3))
        reprs.append(rep_rate(sentence))
        entrs2.append(entropy(types))

    all_entr = None
    total_tokens = sum(lengths)
    for sentence in tqdm(list_of_tokenlists, desc=f'{lc}: computing entropy'):
        if not all_entr:
            all_entr = entropy(all) #total entropy
            for k in all.keys():
                all[k] = 1.0 * all[k] / total_tokens
        entrs.append(entropy_with_est(sentence, all))
    return {
        'types': len(all),
        'tokens': total_tokens,
        'top-10': all.most_common(10),
        'sentences': len(lengths),
        'entropy': all_entr,
        'entropy2': entropy_with_est(all.keys(), all),
        'sent_lens': lengths,
        'sent_entrs': entrs,
        'sent_entrs2': entrs2,
        'sent_ttrs': ttrs,
        'sent_reprs': reprs,
    }


def stats(list_of_tokenlists):
    c = Counter()
    per_sentence = []
    per_sentence_repr = []
    lengths = []
    for sentence in tqdm(list_of_tokenlists, desc='Running statistics'):
        lengths.append(len(sentence))
        c.update(sentence)
        sent_types = set()
        for token in sentence:
            sent_types.add(token)
        per_sentence.append(round(1.0 * len(sent_types) / len(sentence), 3))
        per_sentence_repr.append(rep_rate(sentence))
    ent = entropy(c)
    return {
        'types': len(c),
        'tokens': sum(lengths),
        'top-10': c.most_common(10),
        'sentences': len(lengths),
        'entropy': ent,
        'ttr': round(1.0 * len(c) / sum(lengths), 3),
        'sent_len_min': min(lengths),
        'sent_len_max': max(lengths),
        'sent_len_mean': round(mean(lengths), 4),
        'sent_len_median': median(lengths),
        'sent_ttr_min': min(per_sentence),
        'sent_ttr_max': max(per_sentence),
        'sent_ttr_mean': round(mean(per_sentence), 4),
        'sent_ttr_median': median(per_sentence),
        'sent_repr_min': min(per_sentence_repr),
        'sent_repr_max': max(per_sentence_repr),
        'sent_repr_mean': round(mean(per_sentence_repr), 4),
        'sent_repr_median': median(per_sentence_repr),
    }


def rep_rate(tokenlist):
    rep_count = 0
    for i in range(len(tokenlist) - 1):
        if tokenlist[i] == tokenlist[i + 1]:
            rep_count += 1
    max_reps = 0
    c = Counter(tokenlist)
    for count in c.values():
        max_reps += (count - 1)
    return 0.0 if rep_count == 0 else round(1.0 * rep_count / max_reps, 3)


def get_files(directory, filepattern):
    dir = os.fsencode(directory)
    for file in os.listdir(dir):
        filename = os.fsdecode(file)
        if re.match(filepattern, filename):
            yield os.path.abspath(os.path.join(directory, filename))
    pass


def get_data_robin(dirname):
    robin_data = []
    for tgfile in get_files(dirname, r'.*\.TextGrid'):
        try:
            grid = load_grid(tgfile)
            songs = get_songs_robin(grid)
            robin_data.extend(songs)
        except:
            print(f'Skipped file {tgfile} due to error reading textgrid')
    return robin_data


def data_to_text(data, token_sep, line_sep):
    line_strings = [token_sep.join(x) for x in data]
    return line_sep.join(line_strings)


def load_grid(fname):
    return textgrids.TextGrid(fname)


def get_songs_robin(grid):
    songs = []
    for song in grid['song']:
        if song.text != "":
            songs.append(get_syllables_robin(grid, song.xmin, song.xmax))
    return songs


def get_syllables_robin(grid, start, end):
    syls = []
    for syl in grid['syllable-quality']:
        if syl.text != "" and syl.xmin >= start and syl.xmax <= end:
            syls.append(syl.text.split('-')[0])
    return syls


def get_songs_singletier(grid):
    songs = []
    song = []
    for interval in list(grid.values())[0]:
        if len(song) == 0:
            if interval.text != '':
                song.append(interval.text)
        else:
            if interval.text != '':
                song.append(interval.text)
            elif interval.dur > 3:
                songs.append(song.copy())
                song = []
            else:
                continue
    return songs


def get_data_singletier(dirname, regex=r'.*\.TextGrid'):
    data = []
    for tgfile in get_files(dirname, regex):
        try:
            grid = load_grid(tgfile)
            songs = get_songs_singletier(grid)
            data.extend(songs)
        except:
            print(f'Skipped file {tgfile} due to error reading textgrid')
    return data


def read_textfile(textfile):
    lc = 'unk'
    linedata = []
    try:
        with open(textfile, 'r', encoding='utf-8') as inf:
            for line in inf:
                if line.startswith('# iso'):
                    lc = line.strip().split('\t')[1]
                if line.startswith('<line'):
                    linedata.append(line.strip().split('\t')[1])
    except:
        print(f'Skipped file {textfile}')
    return lc, linedata


def get_data_human(dirname, pattern=r'.*\.txt'):
    data = {}
    for textfile in tqdm(get_files(dirname, pattern), desc='Reading text files'):
        langcode, linelist = read_textfile(textfile)
        if langcode not in data:
            data[langcode] = []
        data[langcode].append(linelist)
    return data


def get_list_of_wordsequences(data, langcode, split=r'\W+'):
    result = []
    for ll in tqdm(data[langcode], desc=f'Tokenizing at word level {langcode}'):
        for line in ll:
            res = re.split(split, line)
            res = list(filter(None, res))
            if len(res) > 0:
                result.append(res)
    return result


def get_list_of_charsequences(listofwordsequences, langcode):
    result = []
    for ll in tqdm(listofwordsequences, desc=f'Tokenizing at word level {langcode}'):
        for word in ll:
            result.append([c for c in word])
    return result


def build_tokenized_data(data, words=True):
    result = {}
    for lc in data:
        result[lc] = {}
        ws = get_list_of_wordsequences(data, lc)
        if words:
            result[lc] = ws
        else:
            result[lc] = get_list_of_charsequences(ws, lc)
    return result


def compute_human_stats(tokenized_data):
    result = {}
    for lc in tokenized_data:
        result[lc] = {}
        result[lc] = stats(tokenized_data[lc])
    return result


def entropy(counter):
    total = counter.total()
    sum = 0.0
    for x in counter.values():
        prob = 1.0 * x / total
        lp = math.log2(prob)
        prod = -prob * lp
        sum += prod
    return sum
    # return round(sum([-x * math.log2( 1.0 * x / total) for x in counter.values()]) / total, 3)


def entropy_with_est(tokens, counter):
    sum = 0.0
    for x in tokens:
        prob = counter[x]
        lp = math.log2(prob)
        prod = -prob * lp
        sum += prod
    return sum
    # return round(sum([-counter[x] * math.log2(counter[x]) for x in tokens]), 3)
