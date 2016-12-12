#!/bin/bash
# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e
set -x

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd $DIR

#download the dataset
echo "Downloading aclImdb..."
#http://ai.stanford.edu/%7Eamaas/data/sentiment/
wget http://ai.stanford.edu/%7Eamaas/data/sentiment/aclImdb_v1.tar.gz

echo "Downloading mosesdecoder..."
#https://github.com/moses-smt/mosesdecoder
wget https://github.com/moses-smt/mosesdecoder/archive/master.zip

#extract package
echo "Unzipping..."
tar -zxvf aclImdb_v1.tar.gz
unzip master.zip

#move train and test set to imdb_data directory 
#in order to process when traing
mkdir -p imdb/train
mkdir -p imdb/test

cp -r aclImdb/train/pos/ imdb/train/pos
cp -r aclImdb/train/neg/ imdb/train/neg

cp -r aclImdb/test/pos/ imdb/test/pos
cp -r aclImdb/test/neg/ imdb/test/neg

#remove compressed package
rm aclImdb_v1.tar.gz
rm master.zip

echo "Done."
