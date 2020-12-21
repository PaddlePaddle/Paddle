#!/bin/bash

# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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


function rocm() {
  # ROCM 3.3 - not work as rocthrust build fail without AMD GPU
  # sed 's#<rocm_repo_version>#3.3#g'  Dockerfile.rocm >test/rocm33.dockerfile
  # sed -ri 's#<rocprim_version>#3.3.0#g' test/rocm33.dockerfile
  # sed -ri 's#<rocthrust_version>#3.3.0#g' test/rocm33.dockerfile
  # sed -ri 's#<hipcub_version>#3.3.0#g' test/rocm33.dockerfile

  # ROCM 3.5
  sed 's#<rocm_repo_version>#3.5.1#g'  Dockerfile.rocm >test/rocm35.dockerfile
  sed -ri 's#<rocprim_version>#3.5.1#g' test/rocm35.dockerfile
  sed -ri 's#<rocthrust_version>#3.5.0#g' test/rocm35.dockerfile
  sed -ri 's#<hipcub_version>#3.5.0#g' test/rocm35.dockerfile

  # ROCM 3.9
  sed 's#<rocm_repo_version>#3.9.1#g'  Dockerfile.rocm >test/rocm39.dockerfile
  sed -ri 's#<rocprim_version>#3.9.0#g' test/rocm39.dockerfile
  sed -ri 's#<rocthrust_version>#3.9.0#g' test/rocm39.dockerfile
  sed -ri 's#<hipcub_version>#3.9.0#g' test/rocm39.dockerfile
}

function main() {
  if [ ! -d "test" ];then
    mkdir test
  fi
  rocm
}

main
