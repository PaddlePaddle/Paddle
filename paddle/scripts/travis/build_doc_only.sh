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

# deploy to remote server
openssl aes-256-cbc -d -a -in $TRAVIS_BUILD_DIR/paddle/scripts/travis/ubuntu.pem.enc -out ubuntu.pem -k $DEC_PASSWD

eval "$(ssh-agent -s)"
chmod 400 ubuntu.pem

ssh-add ubuntu.pem

mkdir -p $TRAVIS_BUILD_DIR/build_docs_versioned/develop
mv $TRAVIS_BUILD_DIR/build_docs/* $TRAVIS_BUILD_DIR/build_docs_versioned/develop/

rsync -r $TRAVIS_BUILD_DIR/build_docs_versioned/ ubuntu@52.76.173.135:/var/content/documentation/

chmod 644 ubuntu.pem
rm ubuntu.pem
