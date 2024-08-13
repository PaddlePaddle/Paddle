#!/usr/bin/env bash

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

# Brief:
#     This code is used for generating the mapping list of Paddle API alias.
#     Only the APIs set with the `DEFINE_ALIAS` flag is enable.
#
# Arguments:
#     None
#
# Usage:
#     Go into the `Paddle` folder and just run `./tools/gen_alias_mapping.sh`
#
# Returns:
#     succ: 0
#
#     Will also print the mapping list to stdout. The format of each line is as below:
#         <real API implement>\t<API recommend>,<API other alias name1>,<API other alias name2>,...


PADDLE_ROOT="$(dirname $(readlink -f ${BASH_SOURCE[0]}))/.."

find ${PADDLE_ROOT}/python/ -name '*.py' \
    | xargs  grep -v '^#' \
    | grep 'DEFINE_ALIAS' \
    | perl -ne '
        if (/\/python\/(.*):from (\.*)(\w.*) import (.*?)\s+#DEFINE_ALIAS\s+$/) {
            my @arr = split(", ", $4);
            foreach $i (@arr) {
                printf "%s|%s|%s|%d\n", $3, $i, substr($1, 0, -3), length($2);
            }
        }' \
    | awk -F '[|/]' '
        {
            key = "";
            val = "";
            if ($2 ~ /.* as .*/) {
                split($2, arr, " as ");
                old = arr[1];
                new = arr[2];
            } else {
                old = $2;
                new = $2;
            }
            for (i = 3; i <= (NF - 1 - $NF); ++i) {
                val = val""$i".";
            }
            val =  val""$1"."old
            for (i = 3; i <= (NF - 1); ++i) {
                if ($i != "__init__") {
                    key = key""$i".";
                }
            }
            key = key""new;
            n2o[key] = val;
        }
        END {
            for (new in n2o) {
                old = n2o[new] in n2o ? n2o[n2o[new]] : n2o[new];
                print old, length(new), new;
            }
        }' \
    | sort -k 1,1 -k 2n,2 \
    | awk '
        {
            o2n[$1] = o2n[$1] ? o2n[$1]","$3 : $3;
        }
        END {
            for (i in o2n) {
                print i"\t"o2n[i];
            }
        }'
