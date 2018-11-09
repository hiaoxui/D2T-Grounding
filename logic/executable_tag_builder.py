from framework import MetaTable
from logic import RWExecutor


def init_tags(meta_tables, diffs):
    """
    Initialize executable tags.
    :param list[MetaTable] meta_tables: All related meta-tables.
    :param list[bool] diffs: Whether to generate diff tags.
    :rtype: list, dict
    """
    tags = list()
    meta_info = {
        'cont': list(),
        'disc': list(),
        'str': list(),
        'diff': list(),
        'copy': list(),
        'all': list(),
        'emit': list(),
        'direct': list(),
    }
    for meta_table in meta_tables:
        meta_info[meta_table.table_name] = list()

    def add_new_tag(meta_table_, field_idx_, type_names, diff):
        """
        Helper function.
        :param meta_table_: Meta information.
        :param int field_idx_: Field index.
        :param list[str] type_names: Keys of meta_info.
        :param bool diff: Property of tags.
        :rtype: None
        """
        tag_idx = len(tags)
        for type_name in type_names:
            meta_info[type_name].append(tag_idx)
        tag = RWExecutor(meta_table_, field_idx_, diff)
        tags.append(tag)

    for meta_table, with_diff in zip(meta_tables, diffs):
        table_name = meta_table.table_name
        for field_idx in meta_table.str_cluster:
            add_new_tag(meta_table, field_idx, ['str', table_name, 'all', 'direct'], False)
        for field_idx in meta_table.disc_cluster:
            add_new_tag(meta_table, field_idx, ['disc', table_name, 'all', 'emit', 'direct'], False)
        for field_idx in meta_table.cont_cluster:
            add_new_tag(meta_table, field_idx, ['cont', 'copy', table_name, 'all', 'emit', 'direct'], False)
        if not with_diff:
            continue
        for field_idx in meta_table.cont_cluster:
            add_new_tag(meta_table, field_idx, ['cont', 'diff', table_name, 'all', 'emit'], True)

    return tags, meta_info
