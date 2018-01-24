# -*- coding: utf-8 -*-
import json
import os

import scrapy
import time
import codecs
import re
from urllib import parse
from PIL import Image


class ZhihuloginSpider(scrapy.Spider):
    name = 'zhihulogin'
    allowed_domains = ['www.zhihu.com']
    start_urls = ['https://www.zhihu.com/']
    start_answer_url = 'https://www.zhihu.com/api/v4/questions/{}/answers?include=data%5B%2A%5D.is_normal%2Cadmin_closed_comment%2Creward_info%2Cis_collapsed%2Cannotation_action%2Cannotation_detail%2Ccollapse_reason%2Cis_sticky%2Ccollapsed_by%2Csuggest_edit%2Ccomment_count%2Ccan_comment%2Ccontent%2Ceditable_content%2Cvoteup_count%2Creshipment_settings%2Ccomment_permission%2Ccreated_time%2Cupdated_time%2Creview_info%2Cquestion%2Cexcerpt%2Crelationship.is_authorized%2Cis_author%2Cvoting%2Cis_thanked%2Cis_nothelp%2Cupvoted_followees%3Bdata%5B%2A%5D.mark_infos%5B%2A%5D.url%3Bdata%5B%2A%5D.author.follower_count%2Cbadge%5B%3F%28type%3Dbest_answerer%29%5D.topics&limit=20&offset=43&sort_by=default%20HTTP/1.1'

    headers = {
        'HOST': 'www.zhihu.com',
        'Referer': 'https://www.zhihu.com',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.94 Safari/537.36'
    }

    def parse(self, response):
        """
        提取出html页面中的所有url 并跟踪这些url进行进一步爬取
        如果提取的url 格式为 ／question/xxx 就下载后直接进入解析函数
        """
        # print(response)

        # 全站搜索过滤法
        all_urls = response.css('a::attr(href)').extract()
        all_urls = [parse.urljoin(response.url, url) for url in all_urls]
        all_urls = filter(lambda x: True if x.startswith("http") else False, all_urls)
        for url in all_urls:
            match_obj = re.match(r'(.*zhihu.com/question/(\d+?))(/|$)', url)
            if match_obj:
                # 如果提取到 question 相关的页面则下载后交由提取函数处理
                question_url = match_obj.group(1)
                question_id = match_obj.group(2)
                yield scrapy.Request(question_url, meta={'question_id': question_id}, headers=self.headers,
                                     callback=self.parse_question)
            else:
                # 如果不是 question 页面则直接进一步跟踪
                yield scrapy.Request(url, headers=self.header, callback=self.parse)

                # # css 提取法
                # question_list = response.css(
                #     '.AnswerItem .ContentItem-title a::attr(href)').extract()
                # # 组合成绝对路径
                # question_list = [parse.urljoin(response.url, url)
                #                  for url in question_list]
                # print(question_list)
                # for url in question_list:
                #     match_obj = re.match(r'(.*zhihu.com/question/(\d+?))(/|$)', url)
                #     # print(match_obj.group(1))
                #     if match_obj:
                #         question_url = match_obj.group(1)
                #         question_id = match_obj.group(2)
                #         yield scrapy.Request(question_url, meta={'question_id': question_id}, headers=self.headers, callback=self.parse_question)
                #     else:
                #         print('匹配 question 失败')

    def parse_question(self, response):
        # 处理 question 页面， 从页面中提取出具体的 question item
        # if "QuestionHeader-title" in response.text:
        # 处理新版本知乎
        # 回答的api: https://www.zhihu.com/api/v4/questions/263413074/answers?limit=20&offset=23
        question_id = response.meta['question_id']
        print(response.url)
        print(question_id)
        yield scrapy.Request(self.start_answer_url.format(question_id), headers=self.headers,
                             callback=self.parse_answer)

    def parse_answer(self, response):
        ans_json = json.loads(response.text)
        is_end = ans_json['paging']['is_end']
        next_url = ans_json['paging']['next']
        answer_id = response.meta['answer_id']
        if not is_end:
            yield scrapy.Request(next_url, headers=self.headers, callback=self.parse_answer)
        yield scrapy.Request(next_url, headers=self.headers, callback=self.parser_comments)


    def parser_comments(self,response):



    def start_requests(self):
        t = str(int(time.time() * 1000))
        captcha_url = 'https://www.zhihu.com/captcha.gif?r=' + t + '&type=login&lang=en'
        return [scrapy.Request(url=captcha_url, headers=self.headers, callback=self.parser_captcha)]

    def parser_captcha(self, response):
        with open('captcha.jpg', 'wb') as f:
            f.write(response.body)
            f.close()
        try:
            im = Image.open('captcha.jpg')
            im.show()
            im.close()
        except:
            print(u'请到 %s 目录找到captcha.jpg 手动输入' % os.path.abspath('captcha.jpg'))
        captcha = input("please input the captcha\n>")
        return scrapy.FormRequest(url='https://www.zhihu.com/#signin', headers=self.headers
                                  , callback=self.login, meta={'captcha': captcha})

    def login(self, response):
        xsrf = str(response.headers['Set-Cookie']).split(';')[0].split('=')[1]
        post_url = 'https://www.zhihu.com/login/phone_num'
        # post_url = 'https://www.zhihu.com/api/v3/oauth/sign_in'
        post_data = {
            "_xsrf": xsrf,
            #废弃的小号  用于测试
            "phone_num": input('user:\n'),
            "password":  input('password:\n'),
            "captcha": response.meta['captcha']
        }
        # return [scrapy.FormRequest(url=post_url, formdata=post_data,
        #                            headers=self.header, callback=self.check_login)]
        yield scrapy.FormRequest(url=post_url, formdata=post_data,
                                   headers=self.headers, callback=self.check_login)

    # 验证返回是否成功
    def check_login(self, response):
        js = json.loads(response.text)
        if 'msg' in js and js['msg'] == '登录成功':
            # for url in self.start_urls:
            #     yield scrapy.Request(url=url, headers=self.header, dont_filter=True
            #                          , callback=self.parse)
            for url in self.start_urls:
                yield scrapy.Request(url=url, headers=self.headers, dont_filter=True
                                     , callback=self.parse)




