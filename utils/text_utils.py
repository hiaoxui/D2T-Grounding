from .ad_hoc import *
from nltk import WordNetLemmatizer
from config import cfg


def extract_tokens(sentence, str_list=None):
    """
    Extract tokens among a sentences, meanwhile picking out the
    proper nouns and numbers.
    :param str sentence: The sentence to tokenize.
    :param list str_list: Proper nouns.
    :rtype: [list, list]
    """
    str_list = str_list or list()
    if len(sentence.split()) < cfg.len_threshold:
        return None
    tokens = []
    slot = []
    sentence += ' '
    sentence.replace('  ', ' ')
    while len(sentence) > 1:
        to_continue = False
        for template_str in str_list:
            template_str += ' '
            if sentence.startswith(template_str):
                slot.append(template_str[:-1])
                tokens.append('<STR>')
                sentence = sentence[len(template_str):]
                to_continue = True
                break
        if to_continue:
            continue

        space_idx = sentence.index(' ')
        next_word = sentence[:space_idx]
        sentence = sentence[space_idx+1:]
        if next_word.isdigit():
            slot.append(int(next_word))
            tokens.append('<NUM>')
            continue

        if next_word.lower() in num_list:
            slot.append(num_list.index(next_word.lower()))
            tokens.append('<NUM>')
            continue

        if len(next_word) > 0:
            tokens.append(next_word)
            slot.append(None)

    return tokens, slot


def collide(l1, l2):
    """
    Detect whether l1 and l2 have common elements.
    :param list l1: List 1.
    :param list l2: List 2.
    :rtype: bool
    """
    return len(set(l1).intersection(l2)) > 0


wnl = WordNetLemmatizer()


def lemmatize(word):
    """
    Helper function of convert.
    :param str word: word to convert.
    :rtype: str
    """
    if word.endswith('ly'):
        word = word[:-2]
    word = wnl.lemmatize(word, 'v')
    word = wnl.lemmatize(word, 'n')
    word = wnl.lemmatize(word, 'a')
    word = wnl.lemmatize(word, 's')
    return word
