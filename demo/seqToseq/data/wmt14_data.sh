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
mkdir wmt14
cd wmt14

# download the dataset
wget http://www-lium.univ-lemans.fr/~schwenk/cslm_joint_paper/data/bitexts.tgz
wget http://www-lium.univ-lemans.fr/~schwenk/cslm_joint_paper/data/dev+test.tgz

# untar the dataset
tar -zxvf bitexts.tgz
tar -zxvf dev+test.tgz
gunzip bitexts.selected/*
mv bitexts.selected train
rm bitexts.tgz
rm dev+test.tgz

# separate the dev and test dataset
mkdir test gen
mv dev/ntst1213.* test
mv dev/ntst14.* gen 
rm -rf dev

set +x
# rename the suffix, .fr->.src, .en->.trg
for dir in train test gen
do 
  filelist=`ls $dir`
  cd $dir
  for file in $filelist
  do 
    if [ ${file##*.} = "fr" ]; then
      mv $file ${file/%fr/src}
    elif [ ${file##*.} = 'en' ]; then
      mv $file ${file/%en/trg}
    fi
  done
  cd ..
done
