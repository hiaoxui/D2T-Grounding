import os
from pprint import pprint


class Config(object):
    def __init__(self, root_dir='./'):
        """
        Configuration.
        :param str root_dir: Root directory.
        """
        self.root_dir = root_dir

        self.k = 2
        self.emit_epoch = 5
        self.trans_epoch = 5

        self.null_mode = 'skip'

        self.use_lemma = True

        # The lowest word frequency
        self.word_freq = 10

        self.len_threshold = 5

        self.train_method = 'ave'
        self.inference_method = 'ave'

        self.null_ratio = 0.4
        self.slack = 0.0
        self.optim_iter = 3

        self.lower_ratio = 0.1

        self.debug = False

        self.use_pr_in_decode = True

        self.no_num = False

        self.jobs = 4

        self.normalized = True

        self.filter = None

        self.no_delta_match = True

        self.double_direction = True

        self.init = 'uniform'

        self.cont_feature = False

    def cache(self, *paths):
        """
        Return path of a file stored/to be stored in cache dir.
        :param str paths: A possible series of folder names plus a filename (necessity).
        :return str: Path.
        """
        path = os.path.join(self.root_dir, 'stuff', 'cache', *paths)

        if len(paths) > 1:
            try:
                os.makedirs(os.path.dirname(path))
            except OSError:
                # Its folder has already created
                pass
        return path

    def data(self, filename):
        """
        Return a file stored in data directory.
        :param str filename: As its name.
        :return str: Path.
        """
        return os.path.join(self.root_dir, 'stuff', 'data', filename)

    def annotate_file(self, filename):
        """

        :param str filename:
        :rtype: str
        """
        return os.path.join(self.root_dir, 'annotate', 'cache', filename)

    def __eq__(self, other):
        """
        Whether two configurations same.
        :param Config other:
        :rtype: bool
        """
        return other.__dict__ == self.__dict__

    def file_name(self, *args):
        """
        Generate file names according to the configuration.
        :param str args: Configuration terms.
        :rtype: str
        """
        ret = list()
        for arg in args[:-1]:
            ret.append(str(getattr(self, arg)))
        ret = '_'.join(ret)
        ret += '.' + args[-1]
        return ret

    def print_info(self):
        pprint(self.__dict__)

    def copy_from(self, other):
        """

        :param Config other:
        :rtype: None
        """
        for key in self.__dict__.keys():
            setattr(self, key, getattr(other, key))


cfg = Config('./')
