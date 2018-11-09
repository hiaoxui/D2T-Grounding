from ..Scenarios import Scenarios


class RotoWire(Scenarios):
    """
    Store the original data for RotoWire.
    """
    def __init__(self, train_bank, valid_bank, test_bank):
        """
        Init.
        :param list train_bank: Contains sentences and tables.
        :param list valid_bank: Contains sentences and tables.
        :param list test_bank: Contains sentences and tables.
        """
        Scenarios.__init__(self)
        self.bank = {
            'train': train_bank,
            'valid': valid_bank,
            'test': test_bank
        }

        # self._build_vocabulary()
        self._record_count()
        self.meta_tables = {
            'LINE': train_bank[0][0]['LINE'].meta_table,
            'BOX': train_bank[0][0]['BOX'].meta_table,
        }
