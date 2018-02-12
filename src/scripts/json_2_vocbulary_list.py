# -*- coding: utf-8 -*-

import json
import jieba

json_path = r'../../corpus/dialog_datas/dialog.json'
sentence_set = r'../../corpus/dialog_datas/sentence_vocbulart.txt'
sentence_dialog = r'../../corpus/dialog_datas/sentence_dialog.txt'

sentences = set()
dialog_set = set()
dialog_set.add('null')

with open(json_path, 'r') as json_file, \
        open(sentence_set, 'w', encoding='utf-8') as set_file, \
        open(sentence_dialog, 'w', encoding='utf-8') as dialog_file:
    for line in json_file.readlines():
        line = line.strip()
        try:
            dialogs = json.loads(line)
            dialog_line = ''
            for sentence in dialogs['dialogs']:
                sentence = sentence.replace('\t', '')
                if sentence not in sentences:
                    sentences.add(sentence)
                    # sentence = sentence.encode('utf-8').decode('unicode_escape')
                    set_file.write(' '.join(jieba.cut(sentence))+'\n')
                if '该评论已删除' in sentence:
                    dialog_line = 'null'
                    break
                cut_sentence = list(jieba.cut(sentence))
                if len(cut_sentence) >= 2:
                    dialog_line += ((' '.join(cut_sentence)) + '\t')

            if dialog_line not in dialog_set:
                dialog_set.add(dialog_line)
                if len(dialog_line.split('\t')) > 1:
                    dialog_file.write(dialog_line + '\n')
                dialog_line = ''
        except Exception as e:
            print(e)
            print(line)
            pass


# todo 这里应该包含各个层级的对话为一行

