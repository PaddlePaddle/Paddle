# 如何贡献代码

我们真诚地感谢您的贡献。你能使用 fork 和 pull request 的工作流来合并（merge）代码。
 
## 代码要求
- 你的代码必须完全遵守 [doxygen](http://www.stack.nl/~dimitri/doxygen/) 的样式。
- 确保编译器选项 WITH\_STYLE\_CHECK 已打开，并且编译器通过代码样式检查。
- 所有代码必须具有单元测试。
- 通过所有单元测试。

以下教程将指导您提交代码。
 
## [Fork](https://help.github.com/articles/fork-a-repo/)
 
转到GitHub页面，然后单击“Fork”按钮。
这就是这么简单。

## 克隆（Clone）

Paddle 目前使用[git流分支模型](http://nvie.com/posts/a-successful-git-branching-model/)。
**develop** 是主分支，其他用户分支是特征分支（feature branches）。

一旦你创建了一个fork，你可以使用你最喜欢的 git 客户端克隆你的仓库（repo）或只是直接在命令行输入：

```shell
# 克隆 fork 到本地
git clone --branch develop https://github.com/USERNAME/Paddle.git
```
如果你的仓库不包含 **develop** 分支，你只需自己创建它。

```shell
git clone https://github.com/USERNAME/Paddle.git Paddle
cd Paddle
git checkout -b develop  # 创建 develop 分支
git remote add upstream https://github.com/PaddlePaddle/Paddle.git  # 添加 upstream 到 baidu/Paddle
git pull upstream develop  # 更新 upstream
git submodule update --init --recursive
```

然后你可以通过做一个本地开发分支开始开发

```shell
git checkout -b MY_COOL_STUFF_BRANCH
```

## 提交（Commit）

提交你的代码：

```shell
# 显示工作树状态
git status
# 添加修改过的文件
git add xx
env EDITOR=vim git commit  # 你可以用 vim/nano/emacs 写下你的注释
```
提交信息的第一行是标题，其他行可以添加一些细节（如果有必要的话）。

## 保持 Fork 状态最新

在拉（pull）你的请求（request）之前，你应该从最新的 PaddlePaddle 同步代码。
为此，你需要首先添加远程（remote）：

```shell
# 观察当前远程仓库配置
git remote -v
# 添加上游（upstream）仓库
git remote add upstream https://github.com/PaddlePaddle/Paddle.git
# 验证新的 upstream
git remote -v
```

用最新的 upstream 更新你的 fork：

```shell
git pull --rebase upstream develop
```
如果本地没有唯一提交，git 将简单地执行快进。但是，如果你一直在做一些改变（绝大多数情况下不应该），你可能要处理冲突。

现在，你的本地主分支与上游修改的一致并是最新的。

## 推送（Push）到 GitHub

```shell
# 在 GitHub 上 push 你的仓库
git push -u origin MY_COOL_STUFF_BRANCH  # 创建远程分支 MY_COOL_STUFF_BRANCH 到 origin.
```

## 拉取请求（Pull Request）

转到 GitHub上 你 fork 的页面，选择你的开发分支并单击 **pull request 按钮**。

## 使用最新版本更新你的 pull 请求

在代码审查（code review）期间，由于 baidu/Paddle 中新的提交导致你的 pull 请求可能会失效。如果没有冲突，GitHub允许自动更新。 你可以点击 pull request 页面中的“更新分支（Update Branch）”按钮。 但是在这种冲突情况下，你需要手动进行更新。你需要在本地仓库执行如下命令：

```shell
git checkout MY_COOL_STUFF_BRANCH
git pull upstream develop
# 你可能需要根据git提示解决冲突
# 创建并测试你的代码
git push origin MY_COOL_STUFF_BRANCH
```
现在你的 Pull Request 是最新的了。

## 修改你的 pull request

当根据审阅者的意见修改 pull 请求时，请使用“git commit”而不是“git commit --amend”来提交更改，以便审阅者可以看到新的请求和旧的请求之间的区别。

可能的命令是

```shell
git checkout MY_COOL_STUFF_BRANCH
git pull upstream develop   # 将本地更新到最新的代码库
# 可能会发生一些冲突
# 开始开发吧！
env EDITOR=vim git commit  # 添加修改日志
git push origin MY_COOL_STUFF_BRANCH
```
