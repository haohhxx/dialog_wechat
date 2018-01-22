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
            r'md.json'
            , 'wb', encoding='utf-8')

    def process_item(self, item, spider):
        line = json.dumps(dict(item)) + '\n'
        # print(line)
        # bline = bytes(line, encoding="utf-8")
        self.file.write(line.decode('unicode-escape'))
        return item
