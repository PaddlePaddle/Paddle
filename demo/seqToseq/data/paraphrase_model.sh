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

dim=32
pretrained_dir='../../model_zoo/embedding/'
preModel=$pretrained_dir'model_'$dim'.emb'
preDict=$pretrained_dir'baidu.dict'

usrDict_dir='pre-paraphrase/'
srcDict=$usrDict_dir'src.dict'
trgDict=$usrDict_dir'trg.dict'

usrModel_dir='paraphrase_model/'
mkdir $usrModel_dir
srcModel=$usrModel_dir'_source_language_embedding'
trgModel=$usrModel_dir'_target_language_embedding'

echo 'extract desired parameters based on user dictionary'
script=$pretrained_dir'extract_para.py'
python $script --preModel $preModel --preDict $preDict \
          --usrModel $srcModel --usrDict $srcDict -d $dim
python $script --preModel $preModel --preDict $preDict \
          --usrModel $trgModel --usrDict $trgDict -d $dim
