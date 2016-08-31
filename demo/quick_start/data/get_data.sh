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

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd $DIR

echo "Downloading Amazon Electronics reviews data..."
# http://jmcauley.ucsd.edu/data/amazon/
wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz

echo "Downloading mosesdecoder..."
#https://github.com/moses-smt/mosesdecoder
wget https://github.com/moses-smt/mosesdecoder/archive/master.zip

unzip master.zip
rm master.zip
echo "Done."
