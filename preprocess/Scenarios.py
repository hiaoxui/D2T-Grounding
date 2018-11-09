from framework import Sentence


class Scenarios(object):
    def __init__(self):
        """
        Abstract constructor.
        """
        self.bank = dict()
        self.n_grams = []
        self.meta_tables = dict()
        self.alignment = dict()

    def _record_count(self):
        """
        Count the number of records.
        :rtype: None
        """
        sample, _ = self.bank['train'][0]
        for table in sample.values():
            table.meta_table.n_records = len(table)

    def extract_sentence(self, phase, limit=999999999):
        """
        Extract all sentences.
        :param str phase: 'train', 'valid' or 'test'
        :param int limit: Upper bound of number of sentences
        :rtype: list[Sentence]
        """
        ret = list()
        for _, sentences in self.bank[phase]:
            ret.extend(sentences)
            if len(ret) > limit:
                break
        ret = ret[:limit]
        return ret
