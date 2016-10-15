#!/bin/bash
# Copyright (c) 2016 Baidu, Inc. All Rights Reserved

cd `dirname $0`

# Add set -e, cd to directory.
set -e
mkdir -p $PWD/../../../build
cd $PWD/../../../build

# Compile Documentation only.
cmake .. -DCMAKE_BUILD_TYPE=Debug -DWITH_GPU=OFF -DWITH_DOC=ON
make paddle_docs paddle_docs_cn

# remove old docs. mv new docs in deeplearning.baidu.com
scp -r doc/html paddle_doc@yq01-idl-gpu-offline42.yq01.baidu.com:/home/paddle_doc/www/doc_new
ssh paddle_doc@yq01-idl-gpu-offline42.yq01.baidu.com "cd ~/www/ && rm -r doc && mv doc_new doc"

scp -r doc_cn/html paddle_doc@yq01-idl-gpu-offline42.yq01.baidu.com:/home/paddle_doc/www/doc_cn_new
ssh paddle_doc@yq01-idl-gpu-offline42.yq01.baidu.com "cd ~/www/ && rm -r doc_cn && mv doc_cn_new doc_cn"
