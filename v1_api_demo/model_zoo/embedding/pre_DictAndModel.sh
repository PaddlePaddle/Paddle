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
BASE_URL='http://paddlepaddle.cdn.bcebos.com/model_zoo/embedding'

DOWNLOAD_ITEMS=(baidu.dict model_32.emb model_64.emb model_128.emb model_256.emb)
ITEM_MD5=(fa03a12321eaab6c30a8fcc9442eaea3
          f88c8325ee6da6187f1080e8fe66c1cd
          927cf70f27f860aff1a5703ebf7f1584
	  a52e43655cd25d279777ed509a1ae27b
	  b92c67fe9ff70fea53596080e351ac80)

for ((i=0; i<${#ITEM_MD5[@]}; i++))
do
  FILENAME=${DOWNLOAD_ITEMS[${i}]}
  REAL_MD5=`wget ${BASE_URL}/${FILENAME} -O - | tee ${FILENAME} | md5sum | cut -d ' ' -f 1`
  EXPECTED_MD5=${ITEM_MD5[${i}]}
  [ "${EXPECTED_MD5}" = "${REAL_MD5}" ]
done
