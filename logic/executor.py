from framework.meta_table import MetaTable
from framework.table import Table


class Executor(object):
    def __init__(self, meta_table):
        """
        Abstract constructor.
        :param MetaTable meta_table: Meta information of tables.
        """
        self.meta_table = meta_table
        self.field_type = None

    def __call__(self, table, index):
        """
        Execute!
        :param Table table: Information source.
        :param int index: Record index.
        """
        raise NotImplementedError
