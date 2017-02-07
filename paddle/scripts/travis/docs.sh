#!/bin/bash

# Add set -e, cd to directory.
source ./common.sh

# Compile Documentation only.
cmake .. -DCMAKE_BUILD_TYPE=Debug -DCMAKE_Fortran_COMPILER=/usr/bin/gfortran-4.8 -DWITH_GPU=OFF -DWITH_DOC=ON ${EXTRA_CMAKE_OPTS}
make paddle_docs paddle_docs_cn

# check websites for broken links
linkchecker doc/en/html/index.html
linkchecker doc/cn/html/index.html

# Parse Github URL
REPO=`git config remote.origin.url`
SSH_REPO=${REPO/https:\/\/github.com\//git@github.com:}
SHA=`git rev-parse --verify HEAD`

# Documentation branch name
# gh-pages branch is used for PaddlePaddle.org. The English version of 
# documentation in `doc` directory, and the chinese version in `doc_cn`
# directory.
TARGET_BRANCH="gh-pages"

# Only deploy master branch to build latest documentation.
SOURCE_BRANCH="master"

# If is not a Github pull request, and in master branch.
if [ "$TRAVIS_PULL_REQUEST" != "false" -o "$TRAVIS_BRANCH" != "$SOURCE_BRANCH"  ]; then
  exit 0
fi

# Clone the repo to output directory
git clone $REPO output
cd output

# checkout github page branch
git checkout $TARGET_BRANCH || git checkout --orphan $TARGET_BRANCH

# remove old docs. mv new docs.
rm -rf doc doc_cn
mv ../doc/cn/html doc_cn
mv ../doc/en/html doc

# Check is there anything changed.
set +e
git diff --exit-code >/dev/null
if [ $? -eq 0 ]; then
  echo "No changes to the output on this push; exiting."
  exit 0
fi
set -e

if [ -n $SSL_KEY ]; then  # Only push updated docs for github.com/PaddlePaddle/Paddle.
  # Commit
  git add .
  git config user.name "Travis CI"
  git config user.email "paddle-dev@baidu.com"
  git commit -m "Deploy to GitHub Pages: ${SHA}"

  # Set ssh private key
  openssl aes-256-cbc -K $SSL_KEY -iv $SSL_IV -in ../../paddle/scripts/travis/deploy_key.enc -out deploy_key -d
  chmod 600 deploy_key
  eval `ssh-agent -s`
  ssh-add deploy_key

  # Push
  git push $SSH_REPO $TARGET_BRANCH

fi
