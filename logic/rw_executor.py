from logic.executor import Executor
from framework.meta_table import MetaTable
from framework.table import Table
from exceptions.type_inconsistent import TypeInconsistent


class RWExecutor(Executor):
    def __init__(self, meta_table, locator, diff=False):
        """
        Initialize an executable tag.
        :param MetaTable meta_table: Meta information of tables.
        :param int locator: The field this executor works on.
        :param bool diff: Whether to output the difference between two records.
        """
        Executor.__init__(self, meta_table)
        self.locator = locator
        self.field_type = meta_table.field_type(self.locator)
        self.diff = diff
        assert not diff or self.field_type == 'cont'

    def __call__(self, table, index):
        """
        Execute.
        :param Table table: Information source.
        :param int/list index: Record index.
        """
        if isinstance(table, dict):
            table = list(table.values())
        if isinstance(table, list):
            for table_ in table:
                if table_.meta_table == self.meta_table:
                    table = table_
                    break
        if not table.meta_table.table_name == self.meta_table.table_name:
            raise TypeInconsistent

        if self.diff:
            lv = table.records[index[0]].values[self.locator]
            rv = table.records[index[1]].values[self.locator]
            return lv - rv
        else:
            return table.records[index].values[self.locator]

    def __str__(self):
        """
        Convert to readable string.
        :rtype: str
        """
        ret = self.meta_table.table_name[0] + '.'
        ret += self.meta_table.idx2field[self.locator]
        if self.diff:
            ret += '.delta'
        return ret
