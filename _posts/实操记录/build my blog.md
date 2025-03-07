---
categories:
  - 实操记录
tags:
	- 搭博客
	- 踩坑
title: build my blog
excerpt: 试一下摘要
cover: https://pic.imgdb.cn/item/64af57971ddac507ccaee683.jpg
date: 2023-07-12
---
### 基本配置
教程，借用llq的博客https://xmnsatay.github.io/2023/02/26/%E8%AE%B0%E5%BD%95%E7%AC%AC%E4%B8%80%E6%AC%A1%E6%90%AD%E5%BB%BA%E4%B8%AA%E4%BA%BA%E5%8D%9A%E5%AE%A2/
### 踩坑
- hexo-d出错
    - 解决：llq说没下载hexo-deployer-git
	    - npm install hexo-deployer-git --save
- 图片只能在本地查看，还不会添加图床
	- 解决：用了聚合图床，后续可能会考虑使用阿里云oss
	- 放个网址[搭建高可移植的Markdown写作和博客环境：Obsidian+PicGo+Hexo+Github | ThinkNotes (cursorhu.github.io)](https://cursorhu.github.io/2022/02/28/%E6%90%AD%E5%BB%BA%E9%AB%98%E5%8F%AF%E7%A7%BB%E6%A4%8D%E7%9A%84Markdown%E5%86%99%E4%BD%9C%E5%92%8C%E5%8D%9A%E5%AE%A2%E7%8E%AF%E5%A2%83%EF%BC%9AObsidian+PicGo+Hexo+Github/)
	- [阿里云OSS PicGo 配置图床教程 超详细 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/104152479)
- 存放这个博客的仓库名字一开始跟github用户名不同
	- 解决：仓库名字改成了用户名
	- 疑问：仓库名字设置为用户名一定可以保证后续博客域名的唯一性吗？域名不区分大小写，然后后面一搜发现github的用户名也不区分大小写哈哈哈🤣
- 成功通过域名访问到博客后，显示的界面只有我的仓库名称
	- 解决：hexo clean后再重新操作，不知道为啥？
- vscode写markdown不丝滑，准备换成obsidian
	- 放个教程[Hexo + Obsidian + Git 完美的博客部署与编辑方案 | EsunR-Blog](https://blog.esunr.xyz/2022/07/e9b42b453d9f.html#3-2-%E4%BD%BF%E7%94%A8-Obsidian-%E6%A8%A1%E6%9D%BF%E5%BF%AB%E9%80%9F%E5%88%9B%E5%BB%BA%E6%96%87%E7%AB%A0)
	- 存在的问题：按照上述方案优化，在ob里创建文件夹后在该文件夹下创建文件，可自动分类补全categories，但是后续想要修改分类的时候会自动变回原来的categories，即修改不了，猜想可能是因为自动分类插件，目前还未解决
- 配置yml文件经常出错
	- 多注意缩进
### 配置主题
放个redefine的官方文档 [Theme Redefine Docs - Redefine Docs (ohevan.com)](https://redefine-docs.ohevan.com/)

### 鼓励自己
搭博客有点麻烦，希望我能好好记录下去，或者说在这里储存储存我的笔记哈哈哈哈😀

    <!--more-->