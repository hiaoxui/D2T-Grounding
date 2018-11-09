import os
import pickle
import time
import random

from preprocess.RotoWire.load_records import load_roto_wire
from estimator import build_vocabulary, to_matrix, Estimator
from logic import init_tags
from utils import *
from .show_alignment import show


class RWTrainer(object):
    def __init__(self, cfg):
        """
        All factors are loaded from configuration.
        """
        self.cfg = cfg
        self.voc = Vocabulary()
        self.unigram = Vocabulary()
        self.tags = list()
        self.meta_info = dict()
        self.sentences = list()

        file_name = self.cfg.file_name('use_lemma', 'word_freq', 'k', 'len_threshold',
                                       'no_delta_match', 'filter', 'preprocessed')
        file_name = self.cfg.cache('scenario', file_name)
        if os.path.exists(file_name):
            with open(file_name, 'rb') as fp:
                self.data = pickle.load(fp)
                self._init_executor()
                self._build_voc()
        else:
            measure('preprocessing')
            self.data = load_roto_wire()
            self._build_voc()
            self._convert_to_matrix()
            self._init_executor()
            self._pre_calculate()
            with open(file_name, 'wb') as fp:
                pickle.dump(self.data, fp)

        self.special_tokens = self.voc[special_tokens]
        self.estimator = Estimator(self.sentences, self.meta_info, len(self.unigram), len(self.voc),
                                   self.special_tokens, self.cfg)

        self.tag_maps = simplify_tags(self.tags)

        measure('End-trainer-init')

    def _build_voc(self):
        """
        Extract sentences to build vocabulary.
        """
        if self.cfg.debug:
            self.sentences = self.data.extract_sentence('train', 1000)
        else:
            self.sentences = self.data.extract_sentence('train')

        if self.cfg.filter is not None:
            self.sentences = self.cluster_sentences(self.sentences)[self.cfg.filter]

        file_name = self.cfg.file_name('use_lemma', 'word_freq', 'k', 'len_threshold', 'debug', 'filter', 'voc')
        file_name = self.cfg.cache('voc', file_name)
        if os.path.exists(file_name):
            with open(file_name, 'rb') as fp:
                self.unigram, self.voc = pickle.load(fp)
                return

        print('Build vocabulary from scratch...')

        self.unigram, self.voc = build_vocabulary(self.sentences)

        with open(file_name, 'wb') as fp:
            pickle.dump([self.unigram, self.voc], fp)

    def _convert_to_matrix(self):
        """
        Convert sentences to matrices.
        :rtype: None
        """
        for phase in ['train', 'valid', 'test']:
            for scenario in self.data.bank[phase]:
                for sentence in scenario[1]:
                    sentence.mat = to_matrix(sentence.token, self.voc)

    def _init_executor(self):
        """
        Initialize executable tags.
        :rtype: None
        """
        meta_tables = list(self.data.meta_tables.values())
        diffs = [False, False]
        if meta_tables[0].table_name == 'LINE':
            diffs[0] = True
        else:
            diffs[1] = True
        self.tags, self.meta_info = init_tags(meta_tables, [True, False])

    def _pre_calculate(self):
        """
        Pre-execute all tags across dataset.
        :rtype: None
        """
        def helper(alignment_):
            """
            Helper function.
            :param list alignment_: Alignment items.
            :rtype: list, list
            """
            lines_ = list()
            boxes_ = list()
            for _0, _1 in alignment_:
                if _0 == 'LINE':
                    lines_.append(_1)
                elif _0 == 'BOX':
                    boxes_.append(_1)
            return lines_, boxes_

        for dataset in self.data.bank.values():
            for tables, sentences in dataset:
                for sentence in sentences:
                    tag_list = sentence.tag = [list() for _ in range(len(self.tags))]
                    match_list = sentence.matches = [list() for _ in range(len(sentence))]
                    line_records, box_records = helper(sentence.alignment)
                    for tag_idx, tag in enumerate(self.tags):
                        if tag_idx in self.meta_info['diff']:
                            if len(line_records) == 2:
                                tag_list[tag_idx].append(tag(tables, line_records))
                        elif tag_idx in self.meta_info['LINE']:
                            for record_idx in line_records:
                                tag_list[tag_idx].append(tag(tables, record_idx))
                        else:
                            for record_idx in box_records:
                                tag_list[tag_idx].append(tag(tables, record_idx))

                        if tag_idx in self.meta_info['disc']:
                            continue
                        if self.cfg.no_delta_match and tag_idx not in self.meta_info['direct']:
                            continue
                        for slot_idx, slot in enumerate(sentence.slot):
                            if slot is not None and slot in tag_list[tag_idx]:
                                match_list[slot_idx].append(tag_idx)

    @staticmethod
    def cluster_sentences(sentences):
        """

        :param list[Sentence] sentences:
        :rtype: dict[str, Sentence]
        """
        clustered = dict()
        for sentence in sentences:
            align = sentence.alignment
            str_ = ''
            for _0, _1 in align:
                str_ += _0[0]
            if str_ not in clustered:
                clustered[str_] = list()
            clustered[str_].append(sentence)
        return clustered

    def train(self):
        """
        Happy training!
        :rtype: None
        """
        measure("Begin Train")
        for e_cnt in range(self.cfg.emit_epoch):
            self.estimator.epoch(self.sentences, False)
        measure("End Emit")
        for e_cnt in range(self.cfg.trans_epoch):
            self.estimator.epoch(self.sentences, True)
        measure("End Trans")
        self.estimator.save()

    def tag_mask(self, tag_indices):
        """

        :param list[int]/int tag_indices:
        :rtype: list[int]/int
        """
        if isinstance(tag_indices, list):
            return [self.tag_mask(item) for item in tag_indices]
        return self.tag_maps[tag_indices]

    def random_test(self):
        key_words = ['']
        sentences = self.data.extract_sentence(phase='test')
        if self.cfg.filter is not None:
            sentences = self.cluster_sentences(sentences)[self.cfg.filter]
        # cfg.normalized = True
        while True:
            key_words = input('keywords (separated with period, q to quit): ')
            key_words = key_words.split('.')

            def adequate(sent_):
                """

                :param Sentence sent_:
                :rtype: bool
                """
                for word in key_words:
                    if word not in sent_.raw:
                        return False
                return True

            if key_words[0] == 'q':
                break
            random.seed(time.time())
            random.shuffle(sentences)

            def select():
                for sent in sentences:
                    if adequate(sent):
                        return sent

            selected = select()
            if selected is None:
                print('not found!')
                continue

            self.estimator.decode([selected])
            show(selected, self.tags)

    def ambiguous(self):
        matches = [0] * 100
        for sentence in self.sentences:
            for s_idx in range(sentence.s):
                if not isinstance(sentence.slot[s_idx], int):
                    continue
                match_ = len(sentence.matches[s_idx])
                matches[match_] += 1
        print(matches)

    def decode_a_sentence(self, sentence):
        """

        :param Sentence sentence:
        :return:
        """
        sentence = self.data.bank['test'][sentence.scenario_idx][1][sentence.sentence_idx]
        self.estimator.decode([sentence])
        ans = sentence.pred
        sentence.pred = None
        return ans
