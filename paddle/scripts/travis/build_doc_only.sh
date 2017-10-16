#!/bin/bash
set -e

rm CMakeLists.txt
mv CMakeLists.doc.txt CMakeLists.txt

# Clean up before new builds.
rm -rf $TRAVIS_BUILD_DIR/build
rm -rf $TRAVIS_BUILD_DIR/build_docs

# Create the build directory for CMake.
mkdir -p $TRAVIS_BUILD_DIR/build
cd $TRAVIS_BUILD_DIR/build

# Compile Documentation only.
cmake .. -DCMAKE_BUILD_TYPE=Debug -DWITH_GPU=OFF -DWITH_MKLDNN=OFF -DWITH_MKLML=OFF -DWITH_DOC=ON
make -j `nproc` gen_proto_py
make -j `nproc` paddle_docs paddle_docs_cn

mkdir -p $TRAVIS_BUILD_DIR/build_docs/en
mkdir -p $TRAVIS_BUILD_DIR/build_docs/cn
mv doc/en/* $TRAVIS_BUILD_DIR/build_docs/en/
mv doc/cn/* $TRAVIS_BUILD_DIR/build_docs/cn/


mkdir -p $TRAVIS_BUILD_DIR/build_docs_versioned/$TRAVIS_BRANCH
mv $TRAVIS_BUILD_DIR/build_docs/* $TRAVIS_BUILD_DIR/build_docs_versioned/$TRAVIS_BRANCH/


# pull PaddlePaddle.org app and strip
# https://github.com/PaddlePaddle/PaddlePaddle.org/archive/master.zip
curl -LOk https://github.com/PaddlePaddle/PaddlePaddle.org/archive/master.zip
unzip master.zip
cd PaddlePaddle.org-master/
cd portal/

sudo pip install -r requirements.txt
mkdir ./tmp
python manage.py deploy_documentation $TRAVIS_BUILD_DIR/build_docs_versioned/$TRAVIS_BRANCH/ $TRAVIS_BRANCH ./tmp

cd ../..

# deploy to remote server
openssl aes-256-cbc -d -a -in $TRAVIS_BUILD_DIR/paddle/scripts/travis/ubuntu.pem.enc -out ubuntu.pem -k $DEC_PASSWD

eval "$(ssh-agent -s)"
chmod 400 ubuntu.pem

ssh-add ubuntu.pem


rsync -r PaddlePaddle.org-master/portal/tmp ubuntu@52.76.173.135:/var/content_staging/docs

rm -rf PaddlePaddle.org-master/
rm -rf master.zip

chmod 644 ubuntu.pem
rm ubuntu.pem
