# -*- coding: utf-8 -*-

import json
import jieba
import re

json_path = r'../../corpus/dialog_datas/dialog.json'
sentence_dialog = r'../../corpus/dialog_datas/sentence_dialog.txt'

line_set = set()

with open(json_path, 'r') as json_file, \
        open(sentence_dialog, 'w', encoding='utf-8') as dialog_file:
    try:
        for line in json_file.readlines():
            dialogs = json.loads(line)
            dialog_line = []
            for sentence in dialogs['dialogs']:
                sentence = sentence.replace('\t', '', 10).strip()
                sentence = sentence.replace('\r', '', 10).strip()
                sentence = sentence.replace('\n', '', 10).strip()
                dr = re.compile(r'<[^>]+>', re.S)
                sentence = dr.sub('', sentence)
                if '该评论已删除' in sentence:
                    continue
                cut_sentence = list(jieba.cut(sentence))
                if len(cut_sentence) >= 1:
                    dialog_line.append((' '.join(cut_sentence)).strip())
            line_str = '\t'.join(dialog_line[:2]) + '\n'
            if line_str not in line_set:
                line_set.add(line_str)
                if len(dialog_line) >= 2:
                    # print(dialog_line)
                    dialog_file.write('\t'.join(dialog_line) + '\n')
    except Exception as e:
        print(e)
        pass


# todo 这里应该包含各个层级的对话为一行

