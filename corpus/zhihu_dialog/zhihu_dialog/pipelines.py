# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html

import json
import codecs


class ZhihuDialogPipeline(object):
    def __init__(self):
        self.file = codecs.open(
            r'../../../corpus/dialog_datas/dialog.json'
            , 'a', encoding='utf-8')
        # self.file = open(r'D:\pycharm_workspace\dialog_wechat\corpus\zhihu_dialog\md.json', 'w')

    def process_item(self, item, spider):
        line = json.dumps(dict(item))
        # line = bytes(line, encoding="utf-32")
        # line = line.decode('unicode-escape')
        # self.file.write(line.replace('\n', '') + '\n')
        self.file.write(line.replace('\n', '') + '\n')
        return item
