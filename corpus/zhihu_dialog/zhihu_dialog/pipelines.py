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
            r'D:\pycharm_workspace\dialog_wechat\corpus\zhihu_dialog\md.json'
            , 'a', encoding='utf-8')
        # self.file = open(r'D:\pycharm_workspace\dialog_wechat\corpus\zhihu_dialog\md.json', 'w')

    def process_item(self, item, spider):
        line = json.dumps(dict(item))
        # print(line)
        bline = bytes(line, encoding="utf-8")
        # print('lineline'+str(line))
        # self.file.write(line.encode('utf-8'))
        self.file.write(bline.decode('unicode-escape')+'\n')
        # self.file.write(line + '\n')
        # self.file.write(str(line.encode('utf-8'))+'\n')
        return item
