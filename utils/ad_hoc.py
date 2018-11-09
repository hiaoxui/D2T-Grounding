import numpy as np
import re


num_list = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven',
            'eight', 'nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen',
            'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen', 'twenty']

special_tokens = ['<num>', '<str>']
special_indices = np.array([1, 0], dtype=np.int32)

key_field = {'LINE': ['TEAM-CITY', 'TEAM-NAME'],
             'BOX': ['PLAYER_NAME', 'SECOND_NAME', 'FIRST_NAME']}

dates = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']

positions = ['F', 'C', 'G', 'N/A']

ppn_errors = {'TOs', 'On', 'City', 'State', 'The', 'New', 'Over', 'This',
              'Mondays', 'In', 'There'}

small_words = ['be', 'the', 'a', 'an', 'and', 'is', 'be', 'are', 'was', 'were', 'to']

pattern_English = re.compile(r'[a-zA-Z ]*')

to_skip = ['(', '-', ')', 'the', ',', '.', '\'', 'n\'t', 'to', '\'s']


def simplify_tags(tags):
    tag2idx = dict()
    for tag_idx, tag in enumerate(tags):
        tag_str = str(tag)
        tag2idx[tag_str] = tag_idx

    tag_map = dict()

    for tag_str, tag_idx in tag2idx.items():
        if '.delta' in tag_str:
            tag_map[tag_idx] = tag2idx[tag_str.replace('.delta', '')]
        elif 'DREB' in tag_str:
            tag_map[tag_idx] = tag2idx[tag_str.replace('DREB', 'REB')]
        elif 'OREB' in tag_str:
            tag_map[tag_idx] = tag2idx[tag_str.replace('OREB', 'REB')]
        else:
            tag_map[tag_idx] = tag_idx

    tag_map[-1] = -1
    tag_map[len(tags)] = -1

    return tag_map
