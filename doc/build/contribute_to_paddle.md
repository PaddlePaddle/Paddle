# Contribute to PaddlePaddle

We sincerely appreciate your contributions. You can use fork and pull request
workflow to merge your code. 
 
## Code Requirements
- Your code mush be fully documented by
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

Once you've created a fork, you can use your favorite git client to clone your
repo or just head straight to the command line:
 
```shell
# Clone your fork to your local machine
git clone git@github.com:USERNAME/paddle.git
```
Then you can start to develop. 

## Commit

Commit your changes by following command lines:

```shell
# show the working tree status
git status
# add modified files
git add xx
git commit -m "commit info"
```
The first line of commit infomation is the title. The second and later lines
are the details if any.

## Keeping Fork Up to Date

Before pull your request, you shold sync you code from the latest Paddle.
To do this, you'll need to add a remote at first:

```shell
# see the current configured remote repository
git remote -v
# add upstream repository
git remote add upstream https://github.com/paddle/paddle.git
# verify the new upstream
git remote -v
```

Update your fork with the latest upstream changes:

```shell
git fetch upstream
git pull upstream master
```

If there are no unique commits locally, git will simply perform a fast-forward.
However, if you have been making changes (in the vast majority of cases you
probably shouldn't be), you may have to deal with conflicts. 

Now, your local master branch is up-to-date with everything modified upstream.

## Push to GitHub

```shell
# push to your repository in Github
git push origin master
```

## Pull Request

Go to the page for your fork on GitHub, select your development branch,
and click the **pull request button**.
