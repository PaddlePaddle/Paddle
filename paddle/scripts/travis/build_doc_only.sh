#!/bin/bash
set -e

rm CMakeLists.txt
mv CMakeLists.doc.txt CMakeLists.txt

rm requirements.txt
cat >> requirements.txt <<EOF
Sphinx==1.5.6
sphinx_rtd_theme==0.1.9
recommonmark
numpy>=1.12
protobuf==3.1
nltk>=3.2.2
rarfile
scipy>=0.19.0
Pillow
EOF

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
mv doc/cn/* $TRAVIS_BUILD_DIR/build_docs/en/

# deploy to remote server
openssl aes-256-cbc -d -a -in $TRAVIS_BUILD_DIR/paddle/scripts/travis/ubuntu.pem.enc -out ubuntu.pem -k $DEC_PASSWD

eval "$(ssh-agent -s)"
chmod 400 ubuntu.pem

ssh-add ubuntu.pem
rsync -r build_docs/ ubuntu@52.76.173.135:/var/content/documentation/develop

chmod 644 ubuntu.pem
rm ubuntu.pem
