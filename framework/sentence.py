import numpy as np


class Sentence(object):
    def __init__(self, raw):
        """
        Sentence contains all the information about a single sentence.
        :param str raw: Raw sentence.
        """
        self.raw = raw
        self.token = list()
        self.slot = list()
        self.alignment = list()
        self.s = -1

        self.tag = list()
        self.mat = np.zeros(0)
        self.matches = list()

        self.pred = list()

        self.scenario_idx = 0
        self.sentence_idx = 0

    def __len__(self):
        """
        The length of the sentence.
        :rtype: int
        """
        return self.s

    def __getitem__(self, idx):
        """
        Get tokens.
        :param int/list[int] idx: Token indices.
        :return str/list[str]:
        """
        if isinstance(idx, list):
            return [self[idx_] for idx_ in idx]
        return self.token[idx]

    def rejoin(self, all_str=False):
        """

        :param bool all_str:
        :rtype: list
        """
        rst = list()
        for idx, item in enumerate(self.slot):
            if item is not None:
                rst.append(item)
            else:
                rst.append(self.token[idx])
        if all_str:
            rst = [str(item) for item in rst]
        return rst
