# Contribute Code

We sincerely appreciate your contributions. You can use fork and pull request
workflow to merge your code.

## Code Requirements
- Your code comments must be fully documented by
  [Doxygen](http://www.stack.nl/~dimitri/doxygen/) style.
- Make sure the compiler option `WITH_STYLE_CHECK` is on and the compiler
  passes the code style check.
- All code must have unit test.
- Pass all unit tests.

The following tutorial guides you into submitting your contibution.

## [Creating a Fork](https://help.github.com/articles/fork-a-repo/)

Just head over to the GitHub page and click the "Fork" button.
It's just that simple.

## Clone

Clone remote repository.

```bash
➜  git clone https://github.com/USERNAME/Paddle
➜  cd Paddle
```

## Create a local branch

Paddle is currently using [Git-flow branching model](http://nvie.com/posts/a-successful-git-branching-model/).

All feature and bug fix development work should be done on a new branch, generally create new branch from `develop` branch .

```bash
➜  git checkout -b my-cool-stuff
```

Before the checkout, you need to keep the current branch directory clean, otherwise the untracked file will be brought to the new branch, which can be inspected by `git status`.

## Using `pre-commit` hook

Paddle developers use [pre-commit](http://pre-commit.com/) tool to manage git
pre-commit hooks. It can help us format source codes (cpp, python), check some
basic thing before commit (only one EOL for each file, do not add a huge file
in git). `pre-commit` tests is a part of unit tests in Travis-CI now, every
PR doesn't fit hook can not be merged into Paddle.

To use [pre-commit](http://pre-commit.com/), you should install it by
`pip install pre-commit`, and currently, Paddle uses `clang-format` to format
c/cpp sources. Please make sure clang-format 3.8+ installed.

Install and run it as follow:

```bash
➜  pip install pre-commit
➜  pre-commit install
```

When you commit your code, the pre-commit hook will check the local code if there is
anything not suitable to commit, and so on.

## Start to develop

In this tutorial, I delete a line in README.md and created a new file.

We can use `git status` to inspect the changes of current directory, `git diff` to see difference.

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
## Build and Test

We package PaddlePaddle's compile environment into a Docker image, called the develop image named `paddle:dev`, it contains all compiling tools that PaddlePaddle needs. 

If you want to build the develop image, just run:

```bash
➜  docker build -t paddle:dev .
```

Then we can use the develop image to build PaddlePaddle source. For example:

```bash
➜  docker run -v $(pwd):/paddle -e "WITH_GPU=OFF" -e "WITH_AVX=ON" -e "WITH_TEST=ON" paddle:dev
```

The above command will compile PaddlePaddle and create a Dockerfile for building production image. All the generated files are in the build directory. "WITH_GPU" controls if the generated production image supports GPU. "WITH_AVX" controls if the generated production image supports AVX. "WITH_TEST" controls if the unit test will be generated.

Then we can generate the production image by copying the compiled PaddlePaddle program into the image by

```bash
➜  docker build -t paddle:prod -f build/Dockerfile .
```

Run unit test finally:

```bash
➜  docker run -it -v $(pwd):/paddle paddle:dev bash -c "cd /paddle/build && ctest"
```

For more details, you can read [this doc](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/getstarted/build_and_install/docker_install_en.rst).

## Commit

Next we cancel the changes to the README.md file and then commit our changes by following command lines:

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

We should write a description of each commit by `git commit` to allow others to know
the changes in these files.

```bash
➜  git commit
CRLF end-lines remover...............................(no files to check)Skipped
yapf.................................................(no files to check)Skipped
Check for added large files..............................................Passed
Check for merge conflicts................................................Passed
Check for broken symlinks................................................Passed
Detect Private Key...................................(no files to check)Skipped
Fix End of Files.....................................(no files to check)Skipped
clang-formater.......................................(no files to check)Skipped
[my-cool-stuff c703c041] add test file
 1 file changed, 0 insertions(+), 0 deletions(-)
 create mode 100644 233
```

## Keeping Fork Up to Date

Before pull your request, you should sync your code from the latest PaddlePaddle.
To do this, you'll need to add a remote at first:

```bash
➜  git remote add upstream https://github.com/PaddlePaddle/Paddle
➜  git remote
origin
upstream
```

Update your fork with the latest upstream changes:

```bash
➜  git fetch upstream
➜  git pull upstream develop
```

Now, your local master branch is up-to-date with everything modified upstream.

## Push to GitHub

```bash
# push to your repository in Github
➜  git push origin my-cool-stuff
```

## Create an issue and a Pull Request

Create an Issue to describe the problem and record its number.

Go to the page for your fork on GitHub, select your development branch,
and click the `New pull request`.

<img width="295" alt="screen shot 2017-04-26 at 9 09 28 pm" src="https://cloud.githubusercontent.com/assets/11692045/25436054/a6d98c66-2ac4-11e7-9cb1-18dd13150230.png">

Then select the target branch:

<img width="750" alt="screen shot 2017-04-26 at 9 11 52 pm" src="https://cloud.githubusercontent.com/assets/11692045/25436139/f83b1e6c-2ac4-11e7-8c0e-add499023c46.png">

We can add `resolve #Issue number` in PR description to close the issue automatically after the PR is merge. More details in <https://help.github.com/articles/closing-issues-via-commit-messages/>.

Then wait for review, if there need to modify, refer to the above steps to update the corresponding origin branch.

## Delete origin branch

After the PR is merge into the main repository, we can delete the remote branch on the PR page.

<img width="775" alt="screen shot 2017-04-26 at 9 18 24 pm" src="https://cloud.githubusercontent.com/assets/11692045/25436457/e4cdd472-2ac5-11e7-9272-badc76c4a23e.png">

Or just run:

```bash
➜  git push origin :my-cool-stuff
```

## Delete local branch

Finally, we delete local branch:

```bash
➜  git checkout develop 

# delete my-cool-stuff branch
➜  git branch -D my-cool-stuff
```
