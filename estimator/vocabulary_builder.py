from utils import *
from config import cfg
from framework import Sentence


def _valid(word_list):
    """

    :param list[str] word_list:
    :rtype: bool
    """
    if len(word_list) == 1:
        return word_list
    word_list = [item.lower() for item in word_list]
    if not all([word.isalpha() for word in word_list]):
        return False
    if collide(word_list, small_words):
        return False
    return True


def build_vocabulary(sentences):
    """
    Build vocabulary from corpus.
    :param list[Sentence] sentences: Tokenized words.
    :rtype: Vocabulary
    """
    uni_gram = Vocabulary(cfg.use_lemma)
    for sentence in sentences:
        sentence = sentence.token
        for token in sentence:
            uni_gram.add([token])

    voc = uni_gram.copy()
    # Maximum length of word spans
    k = cfg.k
    for sentence in sentences:
        sentence = sentence.token
        s = len(sentence)
        for start_idx in range(s):
            for end_idx in range(start_idx+1, min(start_idx+k, s)):
                word_span = sentence[start_idx:end_idx+1]
                if _valid(word_span):
                    lexicon = ' '.join(word_span)
                    voc.add(lexicon)
    voc.trim(word_freq=cfg.word_freq)
    return uni_gram, voc


def build_vocabulary_(sentences):
    """
    Build vocabulary from corpus.
    :param list[Sentence] sentences: Tokenized words.
    :rtype: Vocabulary
    """
    voc = Vocabulary(cfg.use_lemma)
    # Maximum length of word spans
    k = cfg.k
    for sentence in sentences:
        sentence = sentence.token
        s = len(sentence)
        for start_idx in range(s):
            for end_idx in range(start_idx, min(start_idx+1, s)):
                word_span = sentence[start_idx:end_idx+1]
                if _valid(word_span):
                    lexicon = ' '.join(word_span)
                    voc.add(lexicon)
    voc.trim(word_freq=cfg.word_freq)

    for sentence in sentences:
        sentence = sentence.token
        s = len(sentence)
        for start_idx in range(s):
            for end_idx in range(start_idx+1, min(start_idx+k, s)):
                word_span = sentence[start_idx:end_idx+1]
                if _valid(word_span):
                    lexicon = ' '.join(word_span)
                    voc.add(lexicon)
    voc.trim(word_freq=cfg.word_freq)

    return voc


def to_matrix(sentence, voc):
    """
    Convert sentences to matrices.
    :param list[str] sentence: A series of words.
    :param Vocabulary voc: Auxiliary vocabulary.
    :rtype: np.ndarray
    """
    offset = k = cfg.k
    s = len(sentence)
    ret = np.zeros(shape=(s, k), dtype=np.int32)
    sentence = ['<PAD>'] * offset + sentence
    for s_idx in range(s):
        for k_idx in range(k):
            word = ' '.join(sentence[offset+s_idx-k_idx: offset+s_idx+1])
            ret[s_idx, k_idx] = voc[word]
    return ret
