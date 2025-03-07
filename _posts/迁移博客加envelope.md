---
share: "true"
title: 迁移博客加envelope
category: 嘿嘿
---

### 在 Mac 安装 git 和 node.js

使用 brew 进行安装

```text
brew install git
brew install node
```

### 2. 安装 hexo

```text
npm install hexo g
```

这条命令不行（用的时候会出现 `hexo: COMMAND NOT FOUND` 问题）就执行下面这条：

```text
npm install hexo-cli -g
```

### 3. 初始化 hexo 目录

```text
mkdir blog
cd blog
```

### 4. 新建 hexo 服务

```text
hexo init
hexo s
```

打开 `localhost:4000` 查看是否成功。

### 5. 生成 SSH 密钥

先查看本地的 SSH Key：

```text
cd ~/.ssh
```

如果没有，生成一个SSH Key：

```text
ssh-keygen -t rsa -C "example@example.com"
```

最后那个是注册邮箱。

### 6. 关联 GitHub

进入 .ssh 文件夹：

```text
cd ~/.ssh
```

然后打开里面的 `id_rsa.pub` 文件，里面的内容就是 SSH key，复制全部内容。
测试一下是否成功：`ssh git@github.com`

## 二、[配置文件](https://zhida.zhihu.com/search?content_id=181284160&content_type=Article&match_order=1&q=%E9%85%8D%E7%BD%AE%E6%96%87%E4%BB%B6&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NDE0MzM0NjMsInEiOiLphY3nva7mlofku7YiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoxODEyODQxNjAsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.RkSVRshD2qMbDaUBb1IHmlPNeoxKny2EtFKk5nSTzsE&zhida_source=entity)转移

先找到 Windows 下的博客根目录 hexo，复制该目录下的：`_config.yml`、`[scaffolds]、`source`、`themes`、`public`。 再找到 Mac 下的博客根目录 hexo，把刚才复制的内容，直接覆盖替换相同的文件以及文件夹。

## 三、设置个人信息

```text
git config --global user.name "yourname”
git config --global user.email youremail@example.com
```

## 四、发布文章

运行以下命令，查看是否可以成功发布博客：

```text
hexo d -g
```

发现报错：

```text
ERROR Deployer not found: git
```

只需要安装：

```text
npm install hexo-deployer-git --save
```

另外俩报错

 ssh连接方式及Connection closed by 198.xx.xx.xx port 22错误解决
https://blog.csdn.net/m0_60340438/article/details/141273109
Nunjucks Error: 解决方案
https://blog.csdn.net/weixin_45333934/article/details/108274320



然后，图片，test：
![[Pasted image 20250306201923.png|Pasted image 20250306201923.png]]