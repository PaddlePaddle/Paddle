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

# 1. size of pos : neg = 1:1.
# 2. size of testing set = min(25k, len(all_data) * 0.1), others is traning set.
# 3. distinct train set and test set.

set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd $DIR

# Download data
echo "Downloading Amazon Electronics reviews data..."
# http://jmcauley.ucsd.edu/data/amazon/
wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz
echo "Downloading mosesdecoder..."
# https://github.com/moses-smt/mosesdecoder
wget https://github.com/moses-smt/mosesdecoder/archive/master.zip

unzip master.zip
rm master.zip

##################
# Preprocess data 
echo "Preprocess data..."
export LC_ALL=C
UNAME_STR=`uname`

if [ ${UNAME_STR} == 'Linux' ]; then
  SHUF_PROG='shuf'
else
  SHUF_PROG='gshuf'
fi

mkdir -p tmp
python preprocess.py -i reviews_Electronics_5.json.gz
# uniq and shuffle
cd tmp
echo 'Uniq and shuffle...'
cat pos_*|sort|uniq|${SHUF_PROG}> pos.shuffed
cat neg_*|sort|uniq|${SHUF_PROG}> neg.shuffed

min_len=`sed -n '$=' neg.shuffed`
test_num=$((min_len/10))
if [ $test_num -gt 12500 ];then
 test_num=12500
fi
train_num=$((min_len-test_num))

head -n$train_num pos.shuffed >train.pos
head -n$train_num neg.shuffed >train.neg
tail -n$test_num pos.shuffed >test.pos
tail -n$test_num neg.shuffed >test.neg

cat train.pos train.neg | ${SHUF_PROG} >../train.txt
cat test.pos test.neg | ${SHUF_PROG} >../test.txt

cd -
echo 'train.txt' > train.list
echo 'test.txt' > test.list

# use 30k dict
rm -rf tmp
mv dict.txt dict_all.txt
cat dict_all.txt | head -n 30001 > dict.txt
echo 'Done.'
