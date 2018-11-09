from .meta_table import MetaTable
from exceptions import FieldNotFound


class Record(object):
    def __init__(self, table_meta_info, values):
        """
        Constructor of record.
        :param MetaTable table_meta_info: Meta information of tables.
        :param list values: Fields.
        """
        self.meta_table = table_meta_info
        self.values = values
        self.table = None

    def __getitem__(self, item):
        """
        Get the field.
        :param int item: Field index.
        """
        if not item < len(self.values):
            raise FieldNotFound
        return self.values[item]
