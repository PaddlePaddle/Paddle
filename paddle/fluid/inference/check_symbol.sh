#!/bin/sh

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

lib=$1
if [ $# -ne 1 ]; then echo "No input library"; exit -1 ; fi

num_paddle_syms=$(nm -D "${lib}" | grep -c paddle )
num_google_syms=$(nm -D "${lib}" | grep google | grep -v paddle | grep -v phi | grep -v brpc | grep -c "T " )

if [ $num_paddle_syms -le 0 ]; then echo "Have no paddle symbols"; exit -1 ; fi
if [ $num_google_syms -ge 1 ]; then echo "Have some google symbols"; exit -1 ; fi

exit 0
