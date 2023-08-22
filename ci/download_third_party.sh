#!/bin/bash

# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

DATE=$(date +%Y%m%d)
thirdy_party_path=./third_party_${DATE}

third_party_link=https://oss.mthreads.com/mt-ai-data/paddle_musa/third_party.tar.gz
wget --no-check-certificate ${third_party_link} -P ${thirdy_party_path}
tar -zxf ${thirdy_party_path}/third_party.tar.gz
rm -rf ${thirdy_party_path}
