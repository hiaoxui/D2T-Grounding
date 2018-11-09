from tqdm import tqdm
import pickle
import os
import multiprocessing

from HMMs import *
from ._collect import collect_tag_stats, collect_word_freq, collect_possible_values
from utils import *
from framework import Sentence


class Estimator(object):
    def __init__(self, sentences, tag_meta_info, n1, n, copy_indices, cfg):
        """
        Given training data, consisting of anonymized sentences and output of
        executable tags, Estimator will apply EM algorithm along with a simple HMMs
        to optimize the joint probabilities.
        :param list[Sentence] sentences: Corpus.
        where 3 represents STR (a special token that represents proper nouns),
        and 0 represents (another special token that represents numbers)
        :param dict tag_meta_info: The output type of every tags
        :param int n1: The size of unigram
        :param int n: The size of vocabulary
        :param list copy_indices: Special tokens. In the above example, 3 and 0.
        :param str null_mode: Whether to add a special NULL tag. Used in HMMs class.
        """
        null_mode = cfg.null_mode
        self.cfg = cfg
        # Convention: d is the # of examples
        self.d = len(sentences)
        # Convention: k is the maximum of word spans
        self.k = cfg.k

        self.copy_indices = copy_indices.copy()

        # Convention: m is the number of states
        self.m = len(tag_meta_info['all'])
        # Training data

        self.tag_meta_info = tag_meta_info.copy()

        self.null_mode = null_mode

        self.stats = collect_tag_stats(sentences, tag_meta_info, self.cfg)
        self.possible_values = dict()
        self.freq = collect_word_freq(sentences, n)

        # Convention: n is the vocabulary size
        self.n = n
        self.n1 = n1

        if cfg.null_ratio > 0:
            pr = PosteriorRegularization(self.m, self.null_mode, self.k, cfg)
        else:
            pr = None

        self.param = dict()

        self.loaded = True
        if not self.load():
            self.loaded = False
            self.HMMs = HMMs(self.m, null_mode=null_mode, semi=self.k, pr=pr,
                             use_pr_in_decode=cfg.null_ratio > 0)
            self._init_params()

        self.emit_probability = list()

        self.update_HMM = True

    def _init_params(self):
        """
        Initialize parameters for EM before the first epoch.
        :return None: Nothing to return.
        """
        self.param = [None for _ in range(self.m+1)]

        if self.cfg.init == 'uniform':
            word_density = np.ones(shape=(self.n, ), dtype=np.float64) / self.n
        elif self.cfg.init == 'freq':
            word_density = self.freq.copy()
        else:
            raise NotImplemented

        # For continuous values.
        for cont_idx in self.tag_meta_info['cont']:
            self.param[cont_idx] = param_ = dict()
            param_['mean'] = self.stats[cont_idx][0].repeat(self.n)
            param_['variance'] = self.stats[cont_idx][1].repeat(self.n)
            param_['category'] = word_density.copy()
            param_['gross'] = self.stats[cont_idx].copy()
            if self.cfg.cont_feature:
                param_['pre'] = word_density.copy()
                param_['nxt'] = word_density.copy()

        # For discrete values
        for disc_idx in self.tag_meta_info['disc']:
            self.param[disc_idx] = np.outer(self.stats[disc_idx], word_density)

        # For NULL tag
        self.param[-1] = np.zeros(shape=(self.n, ), dtype=np.float64)
        self.param[-1][:] = 1/self.n1

    def _probability(self, sentences):
        """
        Pre-calculate probabilities for training.
        :param list[Sentence] sentences:
        :return list: Probabilities.
        """
        self.possible_values = collect_possible_values(sentences, self.tag_meta_info)

        probability = [None for _ in range(self.m+1)]

        # For continuous values
        for cont_idx in self.tag_meta_info['cont']:
            candidate = dict()
            values = self.possible_values[cont_idx]
            mu, sigma = self.param[cont_idx]['gross']
            # Variables with affix _w belong to individual words
            mu_w = self.param[cont_idx]['mean']
            sigma_w = self.param[cont_idx]['variance']
            nu_w = self.param[cont_idx]['category']
            # Precalculate the emission scores for all possible values
            for value in values:
                # pr: shape [n]
                pr = Gaussian(mu_w, sigma_w, value)
                pr = pr * nu_w
                # pr = pr / Gaussian(mu, sigma, value)
                if self.cfg.normalized:
                    pr = normalize(pr)
                if self.cfg.no_num:
                    pr[:] = 0.
                candidate[value] = pr
                pr[-1] = 0.0
            probability[cont_idx] = candidate

        # For discrete values
        for disc_idx in self.tag_meta_info['disc']:
            probability[disc_idx] = self.param[disc_idx].copy()
            probability[disc_idx][:, -1] = 0

        # For NULL tag
        probability[-1] = self.param[-1].copy()

        # Store it into emit_probability
        self.emit_probability = probability
        return probability

    def _emission(self, sentence, method):
        """
        Calculate the emission score of sentence.
        :param Sentence sentence: Explained above.
        :param str method: How to deal with multiple records?
        :return np.ndarray: Emission scores.
        Shape [s, k, m+1] (including NULL tag)
        """
        # Convention: s is the length of the sentence
        m, n, k, s = self.m, self.n, self.k, sentence.s
        pr = self.emit_probability

        if method == 'ave':
            multiple = np.average
        elif method == 'max':
            multiple = np.max
        elif method == 'sum':
            multiple = np.sum
        else:
            raise Exception

        emit = np.zeros(shape=(s, k, self.m+1), dtype=np.float64)

        for word_idx in range(s):
            single_word = sentence.mat[word_idx, 0]
            if single_word in self.copy_indices:
                emit[word_idx, 0, sentence.matches[word_idx]] = 1.0
                if self.cfg.cont_feature and isinstance(sentence.slot[word_idx], int):
                    pre_word = nxt_word = -1
                    if word_idx > 0:
                        pre_word = sentence.mat[word_idx-1, 0]
                    if word_idx < s-1:
                        nxt_word = sentence.mat[word_idx+1, 0]
                    for m_idx in self.tag_meta_info['cont']:
                        additional = self.param[m_idx]['pre'][pre_word] \
                                * self.param[m_idx]['nxt'][nxt_word]
                        emit[word_idx, 0, m_idx] *= additional
            else:
                token_indices = sentence.mat[word_idx]
                for tag_idx in self.tag_meta_info['emit']:
                    values = sentence.tag[tag_idx]
                    values = [pr[tag_idx][value][token_indices] for value in values]
                    if len(values) == 0:
                        continue
                    probabilities = multiple(values, axis=0)
                    emit[word_idx, :, tag_idx] = probabilities
            if single_word not in special_indices:
                emit[word_idx, 0, -1] = pr[-1][single_word]
            else:
                any_matched = emit[word_idx, 0, :].sum()
                if any_matched == 0:
                    emit[word_idx, 0, -1] = 1e-4

        return emit

    def _blank_counter(self):
        """
        Initialize a blank counter for every epoch.
        :return list: Counter.
        """
        m = self.m

        counter = [np.zeros(0) for _ in range(m)]

        for cont_idx in self.tag_meta_info['cont']:
            counter[cont_idx] = curr_counter = dict()
            curr_counter['center'] = dict()
            for value in self.possible_values[cont_idx]:
                curr_counter['center'][value] = np.zeros(shape=(self.n,), dtype=np.float64)
            if self.cfg.cont_feature:
                curr_counter['pre'] = np.zeros(shape=(self.n,), dtype=np.float64)
                curr_counter['nxt'] = np.zeros(shape=(self.n,), dtype=np.float64)

        for disc_idx in self.tag_meta_info['disc']:
            counter[disc_idx] = dict()
            counter[disc_idx]['center'] = np.zeros_like(self.param[disc_idx], dtype=np.float64)

        # For NULL tag
        null_counter = np.zeros(shape=(self.n,), dtype=np.float64)
        counter.append(null_counter)

        return counter

    def _merge_counter(self, counters):
        big_counter = self._blank_counter()
        for counter in counters:
            for cont_idx in self.tag_meta_info['cont']:
                for value in self.possible_values[cont_idx]:
                    big_counter[cont_idx]['center'][value] += counter[cont_idx]['center'][value]
                if self.cfg.cont_feature:
                    big_counter[cont_idx]['pre'] += counter[cont_idx]['pre']
                    big_counter[cont_idx]['nxt'] += counter[cont_idx]['nxt']

            for disc_idx in self.tag_meta_info['disc']:
                big_counter[disc_idx]['center'] += counter[disc_idx]['center']

            big_counter[self.m] += counter[self.m]

        return big_counter

    def _update_param(self, counter):
        """
        Update emission parameters according to soft counts.
        :param list counter: Soft counter.
        :return None: Nothing to return.
        """
        for cont_idx in self.tag_meta_info['cont']:
            param_ = self.param[cont_idx]
            counter_ = counter[cont_idx]['center']
            if len(counter_) == 0:
                continue
            param_['mean'], param_['variance'] = weighted_mean_std(counter_)
            word_cnt = np.sum(list(counter_.values()), axis=0)
            param_['category'] = normalize(word_cnt)
            for value in counter_:
                counter_[value] = counter_[value].sum()
            if self.cfg.cont_feature:
                counter[cont_idx]['pre'][-1] = 0
                counter[cont_idx]['nxt'][-1] = 0
                param_['pre'] = normalize(counter[cont_idx]['pre'])
                param_['nxt'] = normalize(counter[cont_idx]['nxt'])
                param_['pre'][-1] = 1/self.n
                param_['nxt'][-1] = 1/self.n
            # param_['gross'] = weighted_mean_variance(counter_)

        self._adjust_variance()

        for disc_idx in self.tag_meta_info['disc']:
            if counter[disc_idx]['center'].sum() == 0:
                continue
            self.param[disc_idx] = normalize(counter[disc_idx]['center'])

        # For NULL tag
        m = self.m
        self.param[m] = counter[m] / counter[m].sum()

    def _adjust_variance(self):
        """
        Set a lower bound on variance.
        :rtype: None
        """
        for m_idx in self.tag_meta_info['cont']:
            param_ = self.param[m_idx]
            variance_lower = param_['gross'][1] * self.cfg.lower_ratio
            variances = param_['variance']
            variances[variances < 1.0] = 1.0

    @staticmethod
    def _related_factors():
        return [
            'k',
            'emit_epoch',
            'trans_epoch',
            'null_mode',
            'word_freq',
            'lower_ratio',
            'null_ratio',
            'optim_iter',
            'no_num',
            'debug',
            'normalized',
            'filter',
            'no_delta_match',
            'init',
            'double_direction',
            'cont_feature',
        ]

    def _job(self, arg):
        t_idx, sentences_, config_ = arg
        self.cfg.copy_from(config_)
        m, k = self.m, self.k
        bar = None
        if t_idx == 0:
            bar = tqdm(total=len(sentences_))

        emit_counter_ = self._blank_counter()
        if self.update_HMM:
            hmm_counter_ = self.HMMs.blank_counter()
        else:
            hmm_counter_ = None
        for sentence_ in sentences_:
            if t_idx == 0:
                bar.update()
            emit = self._emission(sentence_, self.cfg.train_method)
            # soft_count: shape [s, k, m+1]
            soft_count = \
                self.HMMs.forward_backward(emit,
                                           cnt=hmm_counter_,
                                           sentence=sentence_,
                                           )

            for s_idx in range(sentence_.s):
                for k_idx in range(k):
                    word_token = sentence_.mat[s_idx, k_idx]
                    # For ground tags
                    for m_idx in self.tag_meta_info['emit']:
                        values = sentence_.tag[m_idx]
                        n_value = len(values)
                        curr_cnt = soft_count[s_idx, k_idx, m_idx]
                        for value in values:
                            emit_counter_[m_idx]['center'][value][word_token] += curr_cnt / n_value
                        if m_idx in self.tag_meta_info['cont']\
                                and self.cfg.cont_feature\
                                and word_token == special_indices[0]\
                                and n_value > 0:
                            pre_word = nxt_word = -1
                            if s_idx > 0:
                                pre_word = sentence_.mat[s_idx-1, 0]
                            if s_idx < sentence_.s-1:
                                nxt_word = sentence_.mat[s_idx+1, 0]
                            emit_counter_[m_idx]['pre'][pre_word] += curr_cnt/n_value
                            emit_counter_[m_idx]['nxt'][nxt_word] += curr_cnt/n_value

                    # For NULL tag
                    emit_counter_[m][word_token] += soft_count[s_idx, k_idx, m]
        if t_idx == 0:
            bar.close()
        return emit_counter_, hmm_counter_

    def epoch(self, sentences, with_transition):
        """
        An EM training epoch.
        :param list[Sentence] sentences:
        :param bool with_transition: Whether to train transition parameters.
        :rtype: None
        """
        self._probability(sentences)
        self.update_HMM = with_transition

        sentences = split_list(sentences, self.cfg.jobs)
        args = list()
        for t_idx, sentences_ in enumerate(sentences):
            args.append([t_idx, sentences_, self.cfg])

        with multiprocessing.Pool(processes=self.cfg.jobs) as pool:
            cnt_ = pool.map(self._job, args)

        emit_cnt = list()
        hmms_cnt = list()
        for emit_cnt_, hmms_cnt_ in cnt_:
            emit_cnt.append(emit_cnt_)
            hmms_cnt.append(hmms_cnt_)

        emit_counter = self._merge_counter(emit_cnt)
        self._update_param(emit_counter)

        if with_transition:
            self.HMMs.update_transition(hmms_cnt)

        measure('An Epoch Is Finished!')

    def decode(self, sentences):
        """

        :param list[Sentence] sentences:
        :rtype: None
        """
        self._probability(sentences)
        for sentence in sentences:
            emit = self._emission(sentence, self.cfg.inference_method)
            trace, max_score = self.HMMs.decode(emit, sentence)
            sentence.pred = trace

    def save(self):
        file_name = self.cfg.file_name(*self._related_factors(), 'param')
        file_name = self.cfg.cache(file_name)
        with open(file_name, 'wb') as fp:
            pickle.dump([self.param, self.HMMs], fp)

    def load(self):
        file_name = self.cfg.file_name(*self._related_factors(), 'param')
        print(file_name)
        file_name = self.cfg.cache(file_name)
        if os.path.exists(file_name):
            print('load param')
            with open(file_name, 'rb') as fp:
                self.param, self.HMMs = pickle.load(fp)
            return True
        else:
            print('init param from scratch')
            return False

    def correlated(self, m_idx):
        """

        :param int m_idx:
        :rtype: list[int]
        """
        if m_idx in self.tag_meta_info['cont']:
            words = self.freq.copy()
            for n_idx in range(self.n):
                words[n_idx] = self.param[m_idx]['category'][n_idx]
            ordered = np.argsort(words)
            return ordered[::-1].tolist()
        elif m_idx in self.tag_meta_info['disc']:
            c = self.param[m_idx].shape[0]
            words = np.outer(np.ones(shape=(c,)), self.freq)
            words = self.param[m_idx] / words
            ordered = np.argsort(words)[:, ::-1].tolist()
            return ordered
