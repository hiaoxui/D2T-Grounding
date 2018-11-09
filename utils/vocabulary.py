from .text_utils import lemmatize


class Vocabulary(object):
    """
    Toolkit.
    """
    def __init__(self, use_lemma=True):
        """
        Initialization.
        :param bool use_lemma: Use lemmatized form or not.
        """
        self.word2idx = {}
        self.idx2word = []
        self.word_counter = {}
        self.distribution = list()
        self.shift = 0
        self.use_lemma = use_lemma
        # A cache of WordNetLemmatizer.
        self.inside_convert = dict()

    def _lemmatize(self, word):
        """
        Convert a word to its lemmatized form.
        :param str/list word: Word to convert.
        """
        if isinstance(word, list):
            return [self._lemmatize(word_) for word_ in word]
        word = word.lower()
        if word not in self.inside_convert:
            self.inside_convert[word] = lemmatize(word)
        return self.inside_convert[word]

    def add(self, new_word, count=1):
        """
        Add new word to vocabulary.
        :param str/list new_word: new word.
        :param int count: Weight.
        """
        if self.use_lemma:
            if isinstance(new_word, str):
                new_word = new_word.split()
            new_word = self._lemmatize(new_word)
            new_word = ' '.join(new_word)
        else:
            if isinstance(new_word, list):
                new_word = ' '.join(new_word)

        if new_word in self.word2idx:
            self.word_counter[new_word] += count
        else:
            self.word_counter[new_word] = count
            self.word2idx[new_word] = len(self.idx2word)
            self.idx2word.append(new_word)

    def _trim(self, size):
        """
        Helper function of trim.
        :param int size: Maximum vocabulary size.
        :return:
        """
        self.idx2word.sort(key=lambda x: -self.word_counter[x])
        cnt = 0
        for idx in range(size, len(self.idx2word)):
            cnt += self.word_counter[self.idx2word[idx]]

        for idx, word in enumerate(self.idx2word):
            self.word2idx[word] = idx
        if size < len(self.idx2word):
            for w in self.idx2word[size:]:
                del self.word2idx[w]
                del self.word_counter[w]
            self.idx2word = self.idx2word[:size]
            self.word_counter[self.idx2word[-1]] += cnt

        self.distribution = [self.word_counter[_] for _ in self.idx2word]

    def trim(self, size=None, word_freq=None):
        """
        Shrink the vocabulary by removing the rare words.
        :param int size: Upper bound of vocabulary size.
        :param int word_freq: Lower bound of word frequency.
        :rtype: None
        """
        size = size or 99999999
        self._trim(size)
        if word_freq is None:
            return
        if word_freq not in self.distribution:
            return
        last_idx = self.distribution.index(word_freq)
        if last_idx == len(self) - 1:
            return
        self._trim(last_idx)

    def __getitem__(self, item):
        """
        Convert a series of words to indices.
        :param str item:
        :return:
        """
        if isinstance(item, list):
            return [self[_] for _ in item]
        if self.use_lemma:
            if isinstance(item, str):
                item = item.split()
            item = self._lemmatize(item)
            item = ' '.join(item)
        else:
            if isinstance(item, list):
                item = ' '.join(item)
        if item in self.word2idx:
            return self.word2idx[item] + self.shift
        else:
            return len(self.word2idx) - 1 + self.shift

    def __call__(self, item):
        """
        Convert an index into word.
        :param int/list item: Word index.
        :rtype: list/str
        """
        if isinstance(item, list):
            return [self(_) for _ in item]
        return self.idx2word[item-self.shift]

    def __len__(self):
        """
        Vocabulary size.
        :rtype: int
        """
        return len(self.word2idx)

    def reset_counter(self):
        """
        Reset word counter.
        :rtype: None
        """
        for w in self.word_counter:
            self.word_counter[w] = 0

    def __contains__(self, item):
        """
        Whether a word has been stored in the vocabulary.
        :param str item: The word to be checked.
        :rtype: bool
        """
        if self.use_lemma:
            if item not in self.inside_convert:
                return False
            item = self.inside_convert[item]
        return item in self.idx2word

    def copy(self):
        """
        Duplicate an identical vocabulary.
        :rtype: Vocabulary
        """
        new_vocabulary = Vocabulary()
        new_vocabulary.use_lemma = self.use_lemma
        new_vocabulary.distribution = self.distribution.copy()
        new_vocabulary.word_counter = self.word_counter.copy()
        new_vocabulary.idx2word = self.idx2word.copy()
        new_vocabulary.word2idx = self.word2idx.copy()
        new_vocabulary.inside_convert = self.inside_convert.copy()
        new_vocabulary.shift = self.shift
        return new_vocabulary
