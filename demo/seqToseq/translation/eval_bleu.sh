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
gen_file=$1
beam_size=$2

# find top1 generating result
top1=$(printf '%s_top1.txt' `basename $gen_file .txt`)
if [ $beam_size -eq 1 ]; then
    awk -F "\t" '{sub(" <e>","",$2);sub(" ","",$2);print $2}' $gen_file >$top1
else
    awk 'BEGIN{
        FS="\t";
        OFS="\t";
        read_pos = 2} {
        if (NR == read_pos){
            sub(" <e>","",$3);
            sub(" ","",$3);
            print $3;
            read_pos += (2 + res_num);
      }}' res_num=$beam_size $gen_file >$top1
fi 

# evalute bleu value
bleu_script=multi-bleu.perl
standard_res=../data/wmt14/gen/ntst14.trg
bleu_res=`perl $bleu_script $standard_res <$top1`

echo $bleu_res
rm $top1
