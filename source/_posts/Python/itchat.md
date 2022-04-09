---
title: "itchat | Log and boot your wechat with python"
ytitle: "itchat | 用python寫一個微信機器人"
description: "itchat是一个开源的微信个人号接口，使用python调用微信从未如此简单。使用不到三十行的代码，你就可以完成一个能够处理所有信息的微信机器人。"
url: itchat
date: 2020/01/22
toc: true
excerpt: "itchat是一个开源的微信个人号接口，使用python调用微信从未如此简单。使用不到三十行的代码，你就可以完成一个能够处理所有信息的微信机器人。"
tags: [Python, HTML]
category: [Python]
cover: 'https://th.bing.com/th/id/R3d9a78ed6fe62aa5ee6e9fd61c092cca?rik=I7LX8qXniM2YLQ&riu=http%3a%2f%2fgetcodify.com%2fwp-content%2fuploads%2f2016%2f10%2fPython_logo.jpg&w=680'
covercopy: '© getcodify.com'
thumbnail: 'https://tse4-mm.cn.bing.net/th/id/OIP.uTOM2B_iUkko5GTxOa3c-wAAAA'
priority: 10000
---


wechat bot<br />微信机器人

确保你可以正常登录网页版微信

```python
import itchat

## 登录
itchat.login()
itchat.auto_login(enableCmdQR=True)
itchat.auto_login(hotReload=True)

## 发送消息
itchat.send(u'你好', 'filehelper')
## 联系人列表
F_list = itchat.get_friends(update=True)


##找到发信息的ID
for i in F_list:
  if i['RemarkName'] == 'XXX': #昵称
  print(i['UserName'])



## friends
friends = itchat.get_friends(update=True)[0:]
## aotu-return
import itchat
##自动回复
@itchat.msg_register(itchat.content.TEXT)
def text_reply(msg):
    return "本人已死，有事招魂，没事烧纸 ಠಗಠ"#+msg["Text"]
##登入
itchat.auto_login()
##保持运行
itchat.run()

```