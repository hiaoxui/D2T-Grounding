from framework import Sentence
from utils import *


def null_b(sent, cfg):
    """
    Generate b for null constraint.
    :param Sentence sent:
    :rtype: float
    """
    b = 0
    for index in sent.mat[:, 0]:
        if index not in special_indices:
            b += 1
    b *= cfg.null_ratio
    return -cfg.null_ratio * sent.s
