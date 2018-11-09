import os
import json
import pandas as pd
from tqdm import tqdm
from nltk import word_tokenize
import pickle
import time

from framework import MetaTable, Table, Sentence
from utils import extract_tokens, Zeller, positions, dates, ppn_errors
from .extract_ppn import extract_ppn
from .roto_wire import RotoWire
from .naiive_alignment import naive_alignment
from config import cfg


def load_roto_wire():
    """
    Load the original text and data.
    :return list: All the scenarios.
    """
    file_name = cfg.file_name('len_threshold', 'raw')
    file_name = cfg.cache('scenario', file_name)
    if os.path.exists(file_name):
        with open(file_name, 'rb') as f:
            return pickle.load(f)

    train_path = cfg.data('train.json')
    valid_path = cfg.data('valid.json')
    test_path = cfg.data('test.json')

    def get_meta_tables(box_record, line_record):
        box_dict = {field_name: 'cont' for field_name in box_record}
        box_dict['FIRST_NAME'] = box_dict['PLAYER_NAME']\
            = box_dict['SECOND_NAME'] = box_dict['TEAM_CITY'] = 'str'
        box_dict['START_POSITION'] = positions[:]

        line_dict = {field_name: 'cont' for field_name in line_record}
        line_dict['TEAM-NAME'] = line_dict['TEAM-CITY'] = 'str'
        line_dict['DATE'] = dates[:]

        return MetaTable('LINE', line_dict), MetaTable('BOX', box_dict)

    j = json.load(open(train_path, 'r', encoding='utf8'))
    line_mt, box_mt = get_meta_tables(j[0]['box_score'].keys(), j[0]['vis_line'].keys())

    def get_str_list(path):
        scenarios = json.load(open(path, 'r', encoding='utf8'))
        names = set()
        all_sentences = []
        for scenario in scenarios:
            names.add(scenario['home_line']['TEAM-CITY'])
            names.add(scenario['home_line']['TEAM-NAME'])
            names.add(scenario['vis_line']['TEAM-CITY'])
            names.add(scenario['vis_line']['TEAM-NAME'])
            box_score = pd.DataFrame(scenario['box_score'])
            summary = ' '.join(scenario['summary'])
            all_sentences.append(' '.join(word_tokenize(summary)))
            for idx in range(len(box_score)):
                record_dict = box_score.iloc[idx].to_dict()
                names.add(record_dict['FIRST_NAME'])
                names.add(record_dict['SECOND_NAME'])
                names.add(record_dict['PLAYER_NAME'])

        other_names = set()
        for sentences in all_sentences:
            for sentence in sentences.split('.'):
                sentence = sentence.strip().split()
                extract_ppn(sentence, other_names, names)

        other_names = other_names.difference(ppn_errors)
        other_names.add('BMO')
        other_names.add('NBA')

        str_list_ = list(other_names.union(names))
        return str_list_

    str_list = get_str_list(train_path)
    str_list.sort(key=lambda str_: -len(str_))

    def helper(path):
        print('loading records...')
        # reserved for elder J
        time.sleep(1)
        scenarios = json.load(open(path, 'r', encoding='utf8'))
        bank = []

        bar = tqdm(total=len(scenarios))
        for scenario_idx, scenario in enumerate(scenarios):
            bar.update()
            line_table = Table(line_mt)
            box_table = Table(box_mt)
            home_line = scenario['home_line']
            home_line['DATE'] = dates[Zeller(scenario['day'])]
            vis_line = scenario['vis_line']
            vis_line['DATE'] = dates[Zeller(scenario['day'])]
            line_table.add(home_line)
            line_table.add(vis_line)
            box_score = pd.DataFrame(scenario['box_score'])
            for idx in range(len(box_score)):
                record_dict = box_score.iloc[idx].to_dict()
                box_table.add(record_dict)
            summary = ' '.join(scenario['summary'])
            summary = word_tokenize(summary)
            summary = ' '.join(summary)
            sentences = []
            sentence_idx = 0
            for sentence in summary.split('.'):
                sent = Sentence(sentence)
                rst = extract_tokens(sentence, str_list)
                if rst is None:
                    continue
                sent.token, sent.slot = rst
                sent.s = len(sent.token)
                sent.sentence_idx = sentence_idx
                sent.scenario_idx = scenario_idx
                sentences.append(sent)
                sentence_idx += 1
            tables_ = {'BOX': box_table, 'LINE': line_table}
            naive_alignment(tables_, sentences)
            bank.append([tables_, sentences])
        bar.close()
        return bank

    train_bank = helper(train_path)
    val_bank = helper(valid_path)
    test_bank = helper(test_path)

    roto_wire = RotoWire(train_bank, val_bank, test_bank)

    with open(file_name, 'wb') as f:
        pickle.dump(roto_wire, f)

    return roto_wire


if __name__ == '__main__':
    rw = load_roto_wire()
