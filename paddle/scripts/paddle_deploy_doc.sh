#!/usr/bin/env bash
# This script should be used only in Travis CI. It will only work after ./paddle_docker_build.sh

echo "TRAVIS_BRANCH: $TRAVIS_BRANCH"

if [ "$TRAVIS_BRANCH" == "develop_doc" ]; then
    PPO_SCRIPT_BRANCH=develop
elif [[ "$TRAVIS_BRANCH" == "develop"  ||  "$TRAVIS_BRANCH" =~ ^v|release/[[:digit:]]+\.[[:digit:]]+(\.[[:digit:]]+)?(-\S*)?$ ]]; then
    PPO_SCRIPT_BRANCH=master
else
    # Early exit, this branch doesn't require documentation build
    exit 0;
fi

echo "Use paddlepaddle.org deploy_docs from: $PPO_SCRIPT_BRANCH branch"

# Fetch the paddlepaddle.org deploy_docs.sh from the appopriate branch
export DEPLOY_DOCS_SH=https://raw.githubusercontent.com/PaddlePaddle/PaddlePaddle.org/$PPO_SCRIPT_BRANCH/scripts/deploy/deploy_docs.sh
export DOCS_DIR=`pwd`

# Pass the python path to capture newly built paddle module
export PYTHONPATH=$PYTHONPATH:$TRAVIS_BUILD_DIR/build/python

echo "Set DOCS_DIR to $DOCS_DIR"
cd ..
curl $DEPLOY_DOCS_SH | bash -s $CONTENT_DEC_PASSWD $TRAVIS_BRANCH $DOCS_DIR $DOCS_DIR/build/doc/
