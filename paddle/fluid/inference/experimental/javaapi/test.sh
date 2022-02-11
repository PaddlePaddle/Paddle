#!/bin/bash

# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

library_path=$1
mkldnn_lib=$library_path"/third_party/install/mkldnn/lib"
mklml_lib=$library_path"/third_party/install/mklml/lib"
paddle_inference_lib=$library_path"/paddle/lib"
export LD_LIBRARY_PATH=$mkldnn_lib:$mklml_lib:$paddle_inference_lib:.
javac -cp $CLASSPATH:JavaInference.jar:. test.java
java -cp $CLASSPATH:JavaInference.jar:. test $2 $3
