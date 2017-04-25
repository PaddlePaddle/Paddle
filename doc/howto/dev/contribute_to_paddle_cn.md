# Git 开发指南 

这个指南将完成一次完整的代码贡献流程。

## Fork

首先 Fork <https://github.com/PaddlePaddle/Paddle>，生成自己目录下的仓库，比如 <https://github.com/USERNAME/Paddle>。

## 克隆（Clone）



## 创建本地分支

所有的 feature 和 bug_fix 的开发工作都应该在一个新的分支上完成，一般从 `develop` 分支上创建新分支。

```bash
# （从当前分支）创建名为 MY_COOL_STUFF_BRANCH 的新分支
➜  git branch MY_COOL_STUFF_BRANCH

# 切换到这个分支上
➜  git checkout MY_COOL_STUFF_BRANCH 
```

也可以通过 `git checkout -b` 一次性创建并切换分支。

```bash
➜  git checkout -b MY_COOL_STUFF_BRANCH
```

值得注意的是，在 checkout 之前，需要保持当前分支目录 clean，否则会把 untracked 的文件也带到新分支上，这可以通过 `git status` 查看。

## 开始开发

在本例中，我删除了 README.md 中的一行，并创建了一个新文件。

通过 `git status` 查看当前状态，这会提示当前目录的一些变化，同时也可以通过 `git diff` 查看文件具体被修改的内容。

```bash
➜  git status
On branch test
Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git checkout -- <file>..." to discard changes in working directory)

	modified:   README.md

Untracked files:
  (use "git add <file>..." to include in what will be committed)

	test

no changes added to commit (use "git add" and/or "git commit -a")
```

## 提交（commit）

接下来我们取消对 README.md 文件的改变，然后提交新添加的 test 文件。

```bash
➜  git checkout -- README.md
➜  git status
On branch test
Untracked files:
  (use "git add <file>..." to include in what will be committed)

	test

nothing added to commit but untracked files present (use "git add" to track)
➜  git add test
```

Paddle 使用 [pre-commit](http://pre-commit.com) 完成代码风格检查的自动化，它会在每次 commit 时自动检查代码是否符合规范，并检查一些基本事宜，因此我们首先安装并在当前目录运行它。

```bash
➜  pip install pre-commit
➜  pre-commit install
```

Git 每次提交代码，都需要写提交说明，这可以让其他人知道这次提交做了哪些改变，这可以通过`git commit -m` 完成。

```bash
➜  git commit -m "add test file"
CRLF end-lines remover...............................(no files to check)Skipped
yapf.................................................(no files to check)Skipped
Check for added large files..............................................Passed
Check for merge conflicts................................................Passed
Check for broken symlinks................................................Passed
Detect Private Key...................................(no files to check)Skipped
Fix End of Files.....................................(no files to check)Skipped
clang-formater.......................................(no files to check)Skipped
[MY_COOL_STUFF_BRANCH c703c041] add test file
 1 file changed, 0 insertions(+), 0 deletions(-)
 create mode 100644 233
```

## 保持本地仓库最新

在准备发起 Pull Request 之前，需要同步原仓库（<https://github.com/PaddlePaddle/Paddle>）最新的代码。

首先通过 `git remote` 查看当前远程仓库的名字。

```bash
➜  git remote
origin
➜  git remote -v
origin	https://github.com/USERNAME/Paddle (fetch)
origin	https://github.com/USERNAME/Paddle (push)
```

这里 origin 是我们 clone 的远程仓库的名字，也就是自己用户名下的 Paddle，接下来我们创建一个原始 Paddle 仓库的远程主机，命名为 upstream。

```bash
➜  git remote add upstream https://github.com/PaddlePaddle/Paddle
➜  git remote
origin
upstream
```

获取 upstream 的最新代码并更新当前分支。

```bash
➜  git fetch upstream
➜  git pull --rebase upstream develop
```

## Push 到远程仓库

将本地的修改推送到 GitHub 上，也就是 https://github.com/USERNAME/Paddle。

```bash
# 推送到远程仓库 origin 的 MY_COOL_STUFF_BRANCH 分支上
➜  git push origin MY_COOL_STUFF_BRANCH
```

## 建立 Issue 并完成 PR

建立一个 Issue 描述问题，记录它的编号。

在 Push 新分支后， https://github.com/USERNAME/Paddle 中会出现新分支提示，点击绿色按钮发起 PR。

![](https://ws1.sinaimg.cn/large/9cd77f2egy1fez1jq9mwdj21js04yq3m.jpg)

选择目标分支：

![](https://ws1.sinaimg.cn/large/9cd77f2egy1fez1ku4a5vj21am04st9l.jpg)

在 PR 的说明中，填写 `solve #Issue编号` 可以在这个 PR 被 merge 后，自动关闭对应的 Issue，具体请见 <https://help.github.com/articles/closing-issues-via-commit-messages/>。

接下来等待 review，如果有需要修改的地方，参照上述步骤更新 origin 中的对应分支即可。

## 删除远程分支

在 PR 被 merge 进主仓库后，我们可以在 PR 的页面删除远程仓库的分支。

![](https://ws1.sinaimg.cn/large/9cd77f2egy1fez1pkqohzj217q05c0tk.jpg)

## 删除本地分支

最后，删除本地分支。

```bash
# 切换到 develop 分支
git checkout develop 

# 删除 MY_COOL_STUFF_BRANCH 分支
git branch -D MY_COOL_STUFF_BRANCH 
```

至此，我们就完成了一次代码贡献的过程。
