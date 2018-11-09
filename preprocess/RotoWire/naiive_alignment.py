from utils import key_field
from framework import Sentence


def naive_alignment(tables, sentences):
    """
    Alignment between sentence and tables, using very naive rules.
    :param dict tables: Corresponding tables.
    :param list[Sentence] sentences: Sentences.
    :rtype: None
    """
    to_align = dict()
    for table_name, table in tables.items():
        table_t = []
        str_cluster = key_field[table_name]
        str_cluster = [table.meta_table.field2idx[field_name] for field_name in str_cluster]
        for record in table.records:
            record_t = []
            for field_idx in str_cluster:
                record_t.append(record[field_idx])
            table_t.append(record_t)
        to_align[table_name] = table_t

    for sentence in sentences:
        alignment = sentence.alignment
        for table_name in ['LINE', 'BOX']:
            for slot in sentence.slot:
                for record_idx, candidate in enumerate(to_align[table_name]):
                    if slot in candidate:
                        attempt = [table_name, record_idx]
                        if attempt in alignment:
                            continue
                        alignment.append(attempt)
