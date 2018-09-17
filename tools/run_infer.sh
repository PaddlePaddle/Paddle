#!/bin/bash

# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
build_dir_name=build.release
cur_dir=$(cd "$(dirname $0)";pwd -P)
build_dir="${cur_dir}/../${build_dir_name}"
export LD_LIBRARY_PATH=${build_dir}/third_party/install/mklml/lib:$LD_LIBRARY_PATH

function usage() {
echo "run_infer.sh ner bs repeat profile test_all_data"
echo "run_infer.sh ner 1 100 on off"
}

#export MKL_NUM_THREADS=1
#export OMP_NUM_THREADS=1
#export OPENBLAS_NUM_THREADS=1
export KMP_AFFINITY="verbose,granularity=fine,compact,0,0" # HT OFF
#export KMP_AFFINITY="verbose,granularity=fine,compact,1,0" # HT ON
#export MKL_VERBOSE=1

bs=1
repeat=1000
use_profile=false
test_all_data=off
num_threads_per_instance=1

exe=
model_name=
all_data=
filter=

if [ ! $1 ]; then
  usage
  exit -1
fi

if [ $1 == "ner" ]; then
  exe="test_analyzer_ner"
  model_name="chinese_ner"
  all_data=""
  filter="Analyzer_Chinese_ner.analysis"
elif [ $1 == "lac" ]; then
  exe="test_analyzer_lac"
  model_name="lac"
  all_data=""
  filter="Analyzer_LAC.analysis"
elif [ $1 == "ocr" ]; then
  exe="test_analyzer_ocr"
  model_name="ocr"
  all_data=""
  filter="Analyzer_vis.analysis"
elif [ $1 == "text" ]; then
  exe="test_analyzer_text_classification"
  model_name="text_classification"
  all_data=""
  filter="text_classification.basic"
elif [ $1 == "rnn1" ]; then
  exe="test_analyzer_rnn1"
  model_name="rnn1"
  all_data=""
  filter="Analyzer.rnn1" 
elif [ $1 == "rnn2" ]; then
  exe="test_analyzer_rnn2"
  model_name="rnn2"
  all_data=""
  filter="Analyzer.rnn2"
else
  usage
  exit -1
fi

if [ $2 ]; then
  bs=$2
fi

if [ $3 ]; then
  repeat=$3
fi

if [ $4 ] && [ $4 == "on" ]; then
  use_profile=true
fi

if [ $5 ] && [ $5 == "on" ]; then
  test_all_data="on"
fi

args="--paddle_num_threads=${num_threads_per_instance} --infer_model=third_party/inference_demo/${model_name}/model --repeat=${repeat} --batch_size=${bs}  --profile=${use_profile} --gtest_filter=${filter}"

if [ ${test_all_data} == "on" ]; then
  args="${args} --test_all_data --infer_data=${all_data}"
else
  args="${args} --infer_data=third_party/inference_demo/${model_name}/data.txt"
fi

echo "command is: ./paddle/fluid/inference/tests/api/${exe} ${args}"
cd ${build_dir}
./paddle/fluid/inference/tests/api/${exe} ${args}
cd -
