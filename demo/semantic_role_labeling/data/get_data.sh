#!/bin/bash
# Copyright (c) 2016 Baidu, Inc. All Rights Reserved
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
wget http://www.cs.upc.edu/~srlconll/conll05st-tests.tar.gz
wget https://www.googledrive.com/host/0B7Q8d52jqeI9ejh6Q1RpMTFQT1k/semantic_role_labeling/verbDict.txt --no-check-certificate
wget https://www.googledrive.com/host/0B7Q8d52jqeI9ejh6Q1RpMTFQT1k/semantic_role_labeling/targetDict.txt --no-check-certificate
wget https://www.googledrive.com/host/0B7Q8d52jqeI9ejh6Q1RpMTFQT1k/semantic_role_labeling/wordDict.txt --no-check-certificate
wget https://www.googledrive.com/host/0B7Q8d52jqeI9ejh6Q1RpMTFQT1k/semantic_role_labeling/emb --no-check-certificate
tar -xzvf conll05st-tests.tar.gz
rm conll05st-tests.tar.gz
cp ./conll05st-release/test.wsj/words/test.wsj.words.gz  .
cp ./conll05st-release/test.wsj/props/test.wsj.props.gz  . 
gunzip test.wsj.words.gz
gunzip test.wsj.props.gz

python extract_pairs.py  -w test.wsj.words -p test.wsj.props -o test.wsj.seq_pair
python extract_dict_feature.py -p test.wsj.seq_pair -f feature 
