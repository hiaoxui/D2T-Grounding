from exceptions import FieldNotFound


class MetaTable(object):

    def __init__(self, table_name, fields):
        """
        Meta information of tables.
        :param str table_name: The name of this table.
        :param dict fields: Field data types.
        """
        self.table_name = table_name

        self.field2idx = dict()
        for idx, name in enumerate(list(fields.keys())):
            self.field2idx[name] = idx
        self.idx2field = [None for _ in self.field2idx]
        for field_name, idx in self.field2idx.items():
            self.idx2field[idx] = field_name

        self.n_field = len(self.idx2field)

        self.cont_cluster, self.disc_cluster, self.str_cluster = [], [], []

        self.cat2idx = dict()
        self.idx2cat = [[] for _ in range(self.n_field)]

        self.n_cat = [-1 for _ in fields]

        self._init_cat2idx(fields)
        self._init_cluster(fields)

        self.stats = [dict() for _ in self.idx2field]

        self.n_records = None

    def __len__(self):
        """
        Number of possible fields.
        :rtype: int
        """
        return len(self.field2idx)

    def _init_cluster(self, fields):
        """
        Initialize meta information.
        :param dict fields: Field data types.
        :rtype: None
        """
        for field_name, field_type in fields.items():
            field_idx = self.field2idx[field_name]
            if isinstance(field_type, list):
                self.disc_cluster.append(field_idx)
            elif field_type == 'cont':
                self.cont_cluster.append(field_idx)
            elif field_type == 'str':
                self.str_cluster.append(field_idx)
            else:
                assert False

    def _init_cat2idx(self, fields):
        """
        Discrete values.
        :param dict fields: Field data types.
        :rtype: None
        """
        for field_name, field_type in fields.items():
            if type(field_type) == list:
                for idx, cat_name in enumerate(field_type):
                    self.cat2idx[field_name+'.'+cat_name] = idx
                    self.idx2cat[self.field2idx[field_name]].append(cat_name)
                self.n_cat[self.field2idx[field_name]] = len(field_type)

    def field_type(self, field_idx):
        """
        Check the data type of fields.
        :param int field_idx: field index.
        :rtype: str
        """
        if isinstance(field_idx, str):
            field_idx = self.field2idx[field_idx]
        if field_idx in self.cont_cluster:
            return 'cont'
        elif field_idx in self.str_cluster:
            return 'str'
        elif field_idx in self.disc_cluster:
            return 'disc'
        raise FieldNotFound

    def convert(self, field_values, fold=1.0):
        """
        Convert a list of field to values.
        :param dict field_values: All the values.
        :param float fold: Weight.
        :rtype: list
        """
        rst = [None for _ in range(len(self.field2idx))]

        for field_name, value in field_values.items():
            assert field_name in self.field2idx
            field_idx = self.field2idx[field_name]
            if self.field_type(field_name) == 'disc':
                rst[field_idx] = self.cat2idx[field_name+'.'+value]
            elif self.field_type(field_name) == 'cont':
                if not value.isdigit():
                    value = 0
                elif isinstance(value, str):
                    value = int(value)
                rst[field_idx] = value
            else:
                rst[field_idx] = value.replace('.', '')

        for field_idx, value in enumerate(rst):
            if value not in self.stats[field_idx]:
                self.stats[field_idx][value] = 0
            self.stats[field_idx][value] += fold

        return rst
