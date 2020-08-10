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

#!/usr/bin/env bash

PADDLE_ROOT="$(dirname $(readlink -f ${BASH_SOURCE[0]}))/.."

find ${PADDLE_ROOT}/python/ -name '*.py' \
    | xargs  grep -v '^#' \
    | grep 'DEFINE_ALIAS' \
    | perl -ne '
        if (/\.\/python\/(.*):from (\.*)(\w.*) import (.*) #DEFINE_ALIAS$/) {
            my @arr = split(", ", $4); 
            foreach $i (@arr) {
                printf "%s|%s|%s|%d\n", $3, $i, substr($1, 0, -3), length($2);
            }
        }' \
    | sort -t '|' -k 2 \
    | awk -F '[|/]' '{
        if ($2 ~ /.* as .*/) {
            split($2, arr, " as ");
            old = arr[1];
            new = arr[2];
        } else {
            old = $2;
            new = $2;
        } 
        for (i = 3; i <= (NF - 1 - $NF); ++i) 
            printf("%s.", $i);
        printf("%s.%s\t", $1, old); 
        for (i = 3; i <= (NF - 1); ++i) {
            if ($i != "__init__")
                printf("%s.", $i);
        } 
        printf("%s\n", new);
    }'
