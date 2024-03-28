# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
rm -rf generated 
# clone triton and reset the specified commit id

triton_repo_directory="triton"
if [ ! -d "$triton_repo_directory" ]; then
    git clone https://github.com/openai/triton.git
    cd triton
    git reset --hard 2217bd2f5c271009f50ab2d2a639bbbb407a2650
    cd -
fi

# these two files are used to compile and link
compile_file=triton/python/triton/tools/compile.py
link_file=triton/python/triton/tools/link.py
gen_paddle_file=generate_custom_paddle_op.py


moe_dir=generated/moe/
mkdir -p ${moe_dir}



python3.8  ${compile_file}  /zhoukangkang/2023-06-06minigpt/PaddleNLP/paddlenlp/transformers/mixtral/modeling_a.py    -n fused_moe_kernel_splitk \
 -o ${moe_dir}/moe     --out-name moe_kernel      \
 -w 4   -ns 2 \
 -s  "*fp16:16, *fp16:16, *fp16:16, \
      *fp16:16, *i32:16, *i32:16, *i32:16, \
      i32:16,i32:16,\
      i32, \
      i32,  \
      i32:16,i32:1,  \
      i32:16,i32:16,i32:1, \
      i32:16,i32:1, \
      i32:1, \
      i32,   \
      16, 64, 128, 11111, 2, 0, 2, 11111" \
      -g   "((EM+16-1)/16) * ((N+64-1)/64), 2, 1" 

python3.8  ${link_file}  ${moe_dir}/*.h -o ${moe_dir}/moe

python3.8 ${gen_paddle_file}  -on "triton_moe" -kn "moe_kernel" -header "moe.h" -oi 2 -cof "generated/moe/triton_moe.cu" -s  "*fp16:16, *fp16:16, *fp16:16, *fp16:16, *i32:16, *i32:16, *i32:16, i32:16,i32:16, i32, i32, i32:16,i32:1, i32:16,i32:16,i32:1,i32:16,i32:1, i32:1, i32, 16, 64, 128, 11111, 2, 0, 2, 11111"




moe_dir=generated/moe2/
mkdir -p ${moe_dir}


python3.8  ${compile_file}  /zhoukangkang/2023-06-06minigpt/PaddleNLP/paddlenlp/transformers/mixtral/modeling_a.py    -n fused_moe_kernel_splitk \
 -o ${moe_dir}/moe2     --out-name moe_kernel2      \
 -w 4   -ns 2 \
 -s  "*fp16:16, *fp16:16, *fp16:16, \
      *fp16:16, *i32:16, *i32:16, *i32:16, \
      i32:16,i32:16,\
      i32, \
      i32,  \
      i32:16,i32:1,  \
      i32:16,i32:16,i32:1, \
      i32:16,i32:1, \
      i32:1, \
      i32,   \
      16, 64, 128, 11111, 2, 1, 1, 11111" \
      -g   "((EM+16-1)/16) * ((N+64-1)/64), 2, 1" 

python3.8  ${link_file}  ${moe_dir}/*.h -o ${moe_dir}/moe2

python3.8 ${gen_paddle_file}  -on "triton_moe2" -kn "moe_kernel2" -header "moe2.h" -oi 2 -cof "generated/moe2/triton_moe2.cu" -s  "*fp16:16, *fp16:16, *fp16:16, *fp16:16, *i32:16, *i32:16, *i32:16, i32:16,i32:16, i32, i32, i32:16,i32:1, i32:16,i32:16,i32:1,i32:16,i32:1, i32:1, i32, 16, 64, 128, 11111, 2, 0, 2, 11111"






# mv the .c file to .cu file
for file in `find ./generated/ -name "*.c"`
do
    mv ${file} ${file}u
done


# 安装triton算子和运行单元测试

python3.8 setup_cuda.py install

