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

#step 1:get kernel registered info
kernel_register_info_file=`mktemp`
PADDLE_ROOT="$( cd "$( dirname "$0" )/../../" && pwd )"
unset GREP_OPTIONS && find ${PADDLE_ROOT}/paddle/phi/kernels -name "*.c*" \
  | xargs sed -e '/PD_REGISTER_\(GENERAL_\)\?KERNEL(/,/)/!d' \
  | awk 'BEGIN { RS="{" }{ gsub(/\n /,""); print $0 }' \
  | grep PD_REGISTER \
  | awk -F ",|\(|\)" '{gsub(/ /,"");$1="";print}' \
  | sort -u  | awk '{gsub(/phi::/,"");gsub(/paddle::platform::/,"");gsub(/dtype::/,"");gsub(/paddle::/,"");print $0}' \
  | grep -v "_grad" > $kernel_register_info_file

#step 2:get simple general inferMeta function wrap info
temp_path=`mktemp -d`
python3 ${PADDLE_ROOT}/python/paddle/utils/code_gen/wrapped_infermeta_gen.py \
  --api_yaml_path ${PADDLE_ROOT}/python/paddle/utils/code_gen/api.yaml \
  --wrapped_infermeta_header_path ${temp_path}/generate.h \
  --wrapped_infermeta_source_path ${temp_path}/generate.cc

find  ${PADDLE_ROOT}/paddle/phi/ -name "*.cc" | xargs grep PD_REGISTER_INFER_META_FN ${temp_path}/generate.cc \
  | awk -F "\(|,|::|\)" '{print $2, $4}' > ${temp_path}/wrap_info.txt

#step 3:get ir's attr_name.
ir_attr_name_info_file=`mktemp`
# phi_cpu attr
all_ir_name=`grep -Eo "PDTCPU_Kernel<.*\"" ${PADDLE_ROOT}/paddle/infrt/dialect/phi/ir/phi_cpu_kernels.td | awk -v FS="<" '{gsub(/\"/,"");print $2}'`
for ir in $all_ir_name
do
  attr_name=`grep "<\"$ir" -A 3 ${PADDLE_ROOT}/paddle/infrt/dialect/phi/ir/phi_cpu_kernels.td  | grep -Eo "Attr:.*)" \
  | awk '{gsub(/F32Attr/,"");gsub(/F64Attr/,"");gsub(/StrAttr/,"");gsub(/BoolAttr/,""); \
  gsub(/SI1Attr/,"");gsub(/SI8Attr/,"");gsub(/SI16Attr/,"");gsub(/SI32Attr/,"");gsub(/SI64Attr/,""); \
  gsub(/UI1Attr/,"");gsub(/UI8Attr/,"");gsub(/I16Attr/,"");gsub(/I32Attr/,"");gsub(/I64Attr/,""); \
  gsub(/I1Attr/,"");gsub(/I8Attr/,"");gsub(/UI16Attr/,"");gsub(/UI32Attr/,"");gsub(/UI64Attr/,""); \
  gsub(/I32ArrayAttr/,"");gsub(/SI32ArrayAttr/,""); \
  gsub(/Attr/,"");gsub(/\)/,""); \
  gsub(/[,:]/,"");print $a}'`
  echo phi_cpu.$ir $attr_name >> $ir_attr_name_info_file
done
# phi_gpu attr
all_ir_name=`grep -Eo "PDTGPU_Kernel<.*\"" ${PADDLE_ROOT}/paddle/infrt/dialect/phi/ir/phi_gpu_kernels.td | awk -v FS="<" '{gsub(/\"/,"");print $2}'`
for ir in $all_ir_name
do
  attr_name=`grep "<\"$ir" -A 3 ${PADDLE_ROOT}/paddle/infrt/dialect/phi/ir/phi_gpu_kernels.td  | grep -Eo "Attr:.*)" \
  | awk '{gsub(/F32Attr/,"");gsub(/F64Attr/,"");gsub(/StrAttr/,"");gsub(/BoolAttr/,""); \
  gsub(/SI1Attr/,"");gsub(/SI8Attr/,"");gsub(/SI16Attr/,"");gsub(/SI32Attr/,"");gsub(/SI64Attr/,""); \
  gsub(/UI1Attr/,"");gsub(/UI8Attr/,"");gsub(/I16Attr/,"");gsub(/I32Attr/,"");gsub(/I64Attr/,""); \
  gsub(/I1Attr/,"");gsub(/I8Attr/,"");gsub(/UI16Attr/,"");gsub(/UI32Attr/,"");gsub(/UI64Attr/,""); \
  gsub(/I32ArrayAttr/,"");gsub(/SI32ArrayAttr/,""); \
  gsub(/Attr/,"");gsub(/\)/,"") \
  gsub(/[,:]/,"");print $a}'`
  echo phi_gpu.$ir $attr_name >> $ir_attr_name_info_file
done

#step 4: merge all infos
#  @input1 => phi kernel infomation : kernel_name kernel_key(GPU/CPU, precision, layout)
#  @input2 => information from api.yaml : kernel_name kernel_function_name inferMeta_function_name 
#  @input3 => information from wrapped_infermeta_gen : ensure the inferMeta function has
#             same signature with kernel function
python3 ${PADDLE_ROOT}/tools/infrt/get_phi_kernel_info.py \
  --paddle_root_path ${PADDLE_ROOT} \
  --kernel_info_file $kernel_register_info_file \
  --infermeta_wrap_file ${temp_path}/wrap_info.txt \
  --attr_info_file $ir_attr_name_info_file \
  --generate_file ${PADDLE_ROOT}/paddle/infrt/kernel/phi/infershaped/infershaped_kernel_launchers.cc
