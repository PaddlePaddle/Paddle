#!/usr/bin/env bash

# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#=================================================
#                   Utils
#=================================================

set -e

EXIT_CODE=0;
tmp_dir=`mktemp -d`

PADDLE_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}")/../../" && pwd )"

unset GREP_OPTIONS && find ${PADDLE_ROOT}/paddle/pten/kernels -name "*.c*" | xargs sed -e '/PT_REGISTER_\(GENERAL_\)\?KERNEL(/,/)/!d' | awk 'BEGIN { RS="{" }{ gsub(/\n /,""); print $0 }' | grep PT_REGISTER | awk -F ",|\(" '{gsub(/ /,"");print $2, $3, $4, $5}' |  sort -u | awk '{gsub(/pten::/,"");print $0}' | grep -v "_grad"
