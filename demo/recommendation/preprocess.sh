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

UNAME_STR=`uname`

if [[ ${UNAME_STR} == 'Linux' ]]; then
	SHUF_PROG='shuf'
else
	SHUF_PROG='gshuf'
fi


cd "$(dirname "$0")"
delimiter='::'
dir=ml-1m
cd data
echo 'generate meta config file'
python config_generator.py config.json > meta_config.json
echo 'generate meta file'
python meta_generator.py $dir meta.bin --config=meta_config.json
echo 'split train/test file'
python split.py $dir/ratings.dat --delimiter=${delimiter} --test_ratio=0.1
echo 'shuffle train file'
${SHUF_PROG} $dir/ratings.dat.train > ratings.dat.train
cp $dir/ratings.dat.test .
echo "./data/ratings.dat.train" > train.list
echo "./data/ratings.dat.test" > test.list
