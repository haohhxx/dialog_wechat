# -*- coding: utf-8 -*-

import json
import jieba

json_path = r'../../corpus/dialog_datas/dialog.json'
sentence_voc = r'../../corpus/dialog_datas/sentence_vocbulart.txt'
sentences = set()
with open(json_path, 'r') as json_file, \
        open(sentence_voc, 'w', encoding='utf-8') as sentence_file:
    for line in json_file.readlines():
        line = line.strip()
        try:
            dialogs = json.loads(line)
            # print(dialogs)
            for sentence in dialogs['dialogs']:
                if sentence not in sentences:
                    sentences.add(sentence)
                    # sentence = sentence.encode('utf-8').decode('unicode_escape')
                    sentence_file.write(' '.join(jieba.cut(sentence))+'\n')
        except Exception as e:
            print(e)
            print(line)
            pass





