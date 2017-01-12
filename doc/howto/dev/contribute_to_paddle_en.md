# Contribute Code

We sincerely appreciate your contributions. You can use fork and pull request
workflow to merge your code.

## Code Requirements
- Your code must be fully documented by
  [doxygen](http://www.stack.nl/~dimitri/doxygen/) style.
- Make sure the compiler option WITH\_STYLE\_CHECK is on and the compiler
  passes the code style check.
- All code must have unit test.
- Pass all unit tests.

The following tutorial guides you into submitting your contibution.

## [Creating a Fork](https://help.github.com/articles/fork-a-repo/)

Just head over to the GitHub page and click the "Fork" button.
It's just that simple.

## Clone

Paddle is currently using [git-flow branching model](http://nvie.com/posts/a-successful-git-branching-model/).
The **develop** is the main branch, and other user's branches are feature branches.

Once you've created a fork, you can use your favorite git client to clone your
repo or just head straight to the command line:

```shell
# Clone your fork to your local machine
git clone --branch develop https://github.com/USERNAME/Paddle.git
```
If your repository doesn't contain **develop** branch, just create it by your own.

```shell
git clone https://github.com/USERNAME/Paddle.git Paddle
cd Paddle
git checkout -b develop  # create develop branch.
git remote add upstream https://github.com/PaddlePaddle/Paddle.git  # add upstream to baidu/Paddle
git pull upstream develop  # update to upstream
```

Then you can start to develop by making a local developement branch

```shell
git checkout -b MY_COOL_STUFF_BRANCH
```

## Using `pre-commit` hook

Paddle developers use [pre-commit](http://pre-commit.com/) tool to manage git
pre-commit hooks. It can help us format source codes (cpp, python), check some
basic thing before commit (only one EOL for each file, do not add a huge file
in git). `pre-commit` tests is a part of unit tests in Travis-CI now, every
PR doesn't fit hook can not be merged into Paddle.

To use [pre-commit](http://pre-commit.com/), you should install it by
`pip install pre-commit`, and currently, Paddle uses `clang-format` to format
c/cpp sources. Please make sure clang-format 3.8+ installed.

Then just run `pre-commit install` in your Paddle clone directory. When you
commit your code, the pre-commit hook will check the local code if there is
anything not suitable to commit, and so on.

## Commit

Commit your changes by following command lines:

```shell
# show the working tree status
git status
# add modified files
git add xx
env EDITOR=vim git commit  # You can write your comments by vim/nano/emacs.
```
The first line of commit infomation is the title. The second and later lines
are the details if any.

## Keeping Fork Up to Date

Before pull your request, you should sync your code from the latest PaddlePaddle.
To do this, you'll need to add a remote at first:

```shell
# see the current configured remote repository
git remote -v
# add upstream repository
git remote add upstream https://github.com/PaddlePaddle/Paddle.git
# verify the new upstream
git remote -v
```

Update your fork with the latest upstream changes:

```shell
git pull --rebase upstream develop
```

If there are no unique commits locally, git will simply perform a fast-forward.
However, if you have been making changes (in the vast majority of cases you
probably shouldn't be), you may have to deal with conflicts.

Now, your local master branch is up-to-date with everything modified upstream.

## Push to GitHub

```shell
# push to your repository in Github
git push -u origin MY_COOL_STUFF_BRANCH  # create remote branch MY_COOL_STUFF_BRANCH to origin.
```

## Pull Request

Go to the page for your fork on GitHub, select your development branch,
and click the **pull request button**.

## Update your pull request with the lastest version

During the code review, your pull request may become stale because new commits in
baidu/Paddle. GitHub allows autmotic update if there is no conflict. You can do this
by clicking the "Update Branch" button in your pull request page. However, in the case
of conflict, you need to do the update manually. You need to do the following on
your local repository:
```shell
git checkout MY_COOL_STUFF_BRANCH
git pull upstream develop
# You may need to resolve the conflict according to the git prompt.
# Make and test your code.
git push origin MY_COOL_STUFF_BRANCH
```
Now your Pull Request is updated with the latest version.

## Revise your pull request

When you revise your pull request according to reviewer's comments, please use 'git commit' instead of 'git commit --amend' to commit your changes so that the reviewers can see the difference between the new pull requrest and the old pull request.

The possible commands are

```shell
git checkout MY_COOL_STUFF_BRANCH
git pull upstream develop   # update local to newest code base.
# May be some conflicts will occured.
# And develop your cool stuff
env EDITOR=vim git commit  # add your revise log
git push origin MY_COOL_STUFF_BRANCH
```
