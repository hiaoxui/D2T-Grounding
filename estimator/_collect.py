from utils import *
from framework import Sentence


def collect_tag_stats(sentences, tag_meta_info, cfg):
    """
    For continuous values, calculate their means and variances.
    For discrete values, calculate their distributions.
    :param list[Sentence] sentences: Including the outputs of executable tags for all the scenarios.
    :param dict tag_meta_info: The output types of each tag.
    :rtype: list[dict]
    """
    m = len(tag_meta_info['all'])
    collection = [dict() for _ in range(m)]

    # Collecting values
    for sentence in sentences:
        for tag_idx in range(m):
            values = sentence.tag[tag_idx].copy()
            if cfg.double_direction and tag_idx in tag_meta_info['diff']:
                values_ = list()
                for value in values:
                    values_.append(value)
                    values_.append(-value)
                values = values_
            for value in values:
                if value not in collection[tag_idx]:
                    collection[tag_idx][value] = 0
                collection[tag_idx][value] += 1

    cont_cluster = tag_meta_info['cont']
    disc_cluster = tag_meta_info['disc']
    str_cluster = tag_meta_info['str']

    # Convert into Gaussian distributions
    for cont_idx in cont_cluster:
        collection[cont_idx] = np.array(weighted_mean_std(collection[cont_idx]))

    # Convert into categorical distributions
    for disc_idx in disc_cluster:
        n_type = len(collection[disc_idx])
        portion = np.zeros(shape=(n_type,), dtype=np.float64)
        for type_idx in range(n_type):
            portion[type_idx] = collection[disc_idx][type_idx]
        portion = portion / portion.sum()
        collection[disc_idx] = portion

    # Convert into categorical distributions
    for str_idx in str_cluster:
        cnt = np.sum(list(collection[str_idx].values()))
        for str_ in collection[str_idx]:
            collection[str_idx][str_] /= cnt

    return collection


def collect_word_freq(sentences, n):
    """
    Calculate the distribution of words.
    :param list[Sentence] sentences: All the sentences for scenarios.
    :param int n: The vocabulary size.
    :return np.ndarray: Word frequencies.
    """
    freq = np.zeros(shape=(n,), dtype=np.float64)
    for sentence in sentences:
        for word in sentence.mat.flatten():
            freq[word] += 1
    freq[-1] = 0
    # Normalization
    freq = normalize(freq)
    return freq


def collect_possible_values(sentences, tag_meta_info):
    """
    Summarize all appeared values for numerical tag.
    :param list[Sentence] sentences: Including the outputs of executable tags for all the scenarios.
    :param dict tag_meta_info: The output types of each tag.
    :rtype: dict
    """
    possible_values = {cont_idx: set() for cont_idx in tag_meta_info['cont']}
    for sentence in sentences:
        for cont_idx in tag_meta_info['cont']:
            values = sentence.tag[cont_idx]
            possible_values[cont_idx] = possible_values[cont_idx].union(values)
    return possible_values
