from .meta_table import MetaTable
from .record import Record


class Table(object):
    def __init__(self, meta_table):
        """
        Table
        :param MetaTable meta_table: Meta information of tables.
        """
        self.meta_table = meta_table
        self.records = []

    def __len__(self):
        """
        Number of records.
        :rtype: int
        """
        return len(self.records)

    def add(self, record):
        """
        Add new record.
        :param Record/dict record: Add new record.
        :rtype: None
        """
        if isinstance(record, dict):
            record = self.meta_table.convert(record)
            record = Record(self.meta_table, record)
        self.records.append(record)
        record.table = self

    def __getitem__(self, item):
        """
        Get record.
        :param int item: Record index.
        :rtype: Record
        """
        return self.records[item]

    def view(self, items):
        """
        A slice of table.
        :param list[int] items:
        :rtype: None/Table
        """
        if len(items) == 0:
            return None
        new_table = Table(self.meta_table)
        for record_idx in items:
            new_table.add(self.records[record_idx])
        return new_table
