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

dir=$1
UT_list=$(ctest -N | awk -F ': ' '{print $2}' | sed '/^$/d' | sed '$d')
for case in $UT_list; do
        flag=$(echo $case|grep -oE '_op')
        if [ -n "$flag" ];then
                if [ -z "$UT_list_prec" ];then
                        UT_list_prec="$case"
                else
                        UT_list_prec="${UT_list_prec}\n${case}"
                fi
        fi
done
echo -e "$UT_list_prec" > "$dir"
