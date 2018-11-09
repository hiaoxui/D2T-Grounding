from framework import Sentence
from logic import RWExecutor


def show(sent, tags):
    """

    :param Sentence sent:
    :param list[RWExecutor] tags:
    :rtype: None
    """
    tokens = sent.rejoin(True)
    for token, tag_idx in zip(tokens, sent.pred):
        if tag_idx >= 0:
            tag_str = str(tags[tag_idx])
        else:
            tag_str = 'NULL'
        print('%25s -- %s' % (token, tag_str))
