# -*- coding: utf-8 -*-

import requests

headers = {
        'accept': 'application/json, text/plain, */*'
        # , 'Accept-Encoding': 'gzip, deflate, br'
        , 'Accept-Language': 'zh-CN,zh;q=0.9'
        , 'authorization': 'Bearer 2|1:0|10:1516509952|4:z_c0|92:Mi4xeXdiRUFRQUFBQUFBa0VCeUdNLW9DaVlBQUFCZ0FsVk5fMmhSV3dBYWM4NmxoeXo0Z0dyckFsbVIzeUdhS09oeWtR|d26d66b10b606e6255352643b05f32979c80c94498ea70dd5ee6711ebc970218'
        , 'Connection': 'keep-alive'
        , 'Cookie': '"d_c0="AJBAchjPqAqPTkzZHIOLxPXFDkfhm9Oriw8=|1475907269"; _za=463755ad-d993-44dc-82e4-47f6061b6b8c; _zap=851b294d-2d2c-4e27-bf63-b09c0b22cc3a; q_c1=41f2eb9cc3514af090045b8b30b89d93|1503584708000|1475907269000; __utma=51854390.247998483.1486386912.1511178218.1514697087.21; __utmz=51854390.1514697087.21.21.utmcsr=baidu|utmccn=(organic)|utmcmd=organic; __utmv=51854390.000--|2=registration_date=20150608=1^3=entry_date=20161008=1; r_cap_id="YjA5ZmUzMjFlOWIxNDY2M2EzN2FmNzgxZThhNjlmNjM=|1516108673|eff33ea1b3c5fc64735626daaf5a3e16fb031ebe"; cap_id="OTIzZWJhMDM5YmE5NGZhN2IyZDg3MzMzZmZiY2U4NTI=|1516108673|32d736e819e354a44b74a9d8da9f5278e93ba343"; l_cap_id="YTY3MThmYjk4ZTM2NDZiYzkxY2QyODYwMzYzNDQzNzU=|1516108673|4e53b75d24d91ae0d3fdb90e723491518266d79d"; client_id="QzdGRjg5RERDRkVGQkU2RUJEQzAzQzYwQjYzMEMwNzM=|1516108677|4f4579f0618b2938b66e01df26b739e15ce3f4ce"; _xsrf=013a32c8-07b6-4127-89e0-d381092cee4e"'
        , 'Host': 'www.zhihu.com'
        , 'Referer': 'https://www.zhihu.com/'
        , 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36'
        , 'X-UDID': 'AJBAchjPqAqPTkzZHIOLxPXFDkfhm9Oriw8='
}

url_str = 'https://www.zhihu.com/api/v4/comments/388124228/conversation?include=%24%5B*%5D.author%2Creply_to_author%2Ccontent%2Cvote_count'
response = requests.get(url_str, headers=headers)
print(response.content)
# print(str(response.content, encoding='utf-8'))
print(response.content.decode(encoding='utf-8'))
