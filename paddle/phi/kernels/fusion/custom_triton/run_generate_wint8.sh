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
triton_op_file=/zhoukangkang/triton/python/paddle_tutorials/weight-only-int8.py

wint8_dir=generated/wint8/
mkdir -p ${wint8_dir}


# -n : the kernel name decorated by @triton.jit in your py file
# -o : the output file name 
# --out-name : the name of the kernel in c++ for your invoke





python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/weight-only-int8.py    -n wint8_kernel   -o ${wint8_dir}/wint8     --out-name wint8_kernel     -w 4   -ns 2 -s   "*fp16:16, *u8:16, *fp16:16, *fp16:16, *fp16:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:1,i32:16, i32:16,i32:1, 16, 64, 64, 1, 1" -g   "((M+16-1)/16) * ((N+64-1)/64), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/weight-only-int8.py    -n wint8_kernel   -o ${wint8_dir}/wint8     --out-name wint8_kernel     -w 4   -ns 2 -s   "*fp16:16, *u8:16, *fp16:16, *fp16:16, *fp16:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:1,i32:16, i32:16,i32:1, 16, 64, 64, 1, 2" -g   "((M+16-1)/16) * ((N+64-1)/64), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/weight-only-int8.py    -n wint8_kernel   -o ${wint8_dir}/wint8     --out-name wint8_kernel     -w 4   -ns 2 -s   "*fp16:16, *u8:16, *fp16:16, *fp16:16, *fp16:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:1,i32:16, i32:16,i32:1, 16, 64, 64, 1, 4" -g   "((M+16-1)/16) * ((N+64-1)/64), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/weight-only-int8.py    -n wint8_kernel   -o ${wint8_dir}/wint8     --out-name wint8_kernel     -w 4   -ns 2 -s   "*fp16:16, *u8:16, *fp16:16, *fp16:16, *fp16:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:1,i32:16, i32:16,i32:1, 16, 64, 64, 1, 8" -g   "((M+16-1)/16) * ((N+64-1)/64), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/weight-only-int8.py    -n wint8_kernel   -o ${wint8_dir}/wint8     --out-name wint8_kernel     -w 4   -ns 2 -s   "*fp16:16, *u8:16, *fp16:16, *fp16:16, *fp16:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:1,i32:16, i32:16,i32:1, 16, 64, 128, 1, 1" -g   "((M+16-1)/16) * ((N+64-1)/64), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/weight-only-int8.py    -n wint8_kernel   -o ${wint8_dir}/wint8     --out-name wint8_kernel     -w 4   -ns 2 -s   "*fp16:16, *u8:16, *fp16:16, *fp16:16, *fp16:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:1,i32:16, i32:16,i32:1, 16, 64, 128, 1, 2" -g   "((M+16-1)/16) * ((N+64-1)/64), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/weight-only-int8.py    -n wint8_kernel   -o ${wint8_dir}/wint8     --out-name wint8_kernel     -w 4   -ns 2 -s   "*fp16:16, *u8:16, *fp16:16, *fp16:16, *fp16:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:1,i32:16, i32:16,i32:1, 16, 64, 128, 1, 4" -g   "((M+16-1)/16) * ((N+64-1)/64), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/weight-only-int8.py    -n wint8_kernel   -o ${wint8_dir}/wint8     --out-name wint8_kernel     -w 4   -ns 2 -s   "*fp16:16, *u8:16, *fp16:16, *fp16:16, *fp16:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:1,i32:16, i32:16,i32:1, 16, 64, 128, 1, 8" -g   "((M+16-1)/16) * ((N+64-1)/64), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/weight-only-int8.py    -n wint8_kernel   -o ${wint8_dir}/wint8     --out-name wint8_kernel     -w 4   -ns 2 -s   "*fp16:16, *u8:16, *fp16:16, *fp16:16, *fp16:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:1,i32:16, i32:16,i32:1, 16, 64, 256, 1, 1" -g   "((M+16-1)/16) * ((N+64-1)/64), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/weight-only-int8.py    -n wint8_kernel   -o ${wint8_dir}/wint8     --out-name wint8_kernel     -w 4   -ns 2 -s   "*fp16:16, *u8:16, *fp16:16, *fp16:16, *fp16:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:1,i32:16, i32:16,i32:1, 16, 64, 256, 1, 2" -g   "((M+16-1)/16) * ((N+64-1)/64), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/weight-only-int8.py    -n wint8_kernel   -o ${wint8_dir}/wint8     --out-name wint8_kernel     -w 4   -ns 2 -s   "*fp16:16, *u8:16, *fp16:16, *fp16:16, *fp16:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:1,i32:16, i32:16,i32:1, 16, 64, 256, 1, 4" -g   "((M+16-1)/16) * ((N+64-1)/64), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/weight-only-int8.py    -n wint8_kernel   -o ${wint8_dir}/wint8     --out-name wint8_kernel     -w 4   -ns 2 -s   "*fp16:16, *u8:16, *fp16:16, *fp16:16, *fp16:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:1,i32:16, i32:16,i32:1, 16, 64, 256, 1, 8" -g   "((M+16-1)/16) * ((N+64-1)/64), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/weight-only-int8.py    -n wint8_kernel   -o ${wint8_dir}/wint8     --out-name wint8_kernel     -w 4   -ns 2 -s   "*fp16:16, *u8:16, *fp16:16, *fp16:16, *fp16:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:1,i32:16, i32:16,i32:1, 16, 128, 64, 1, 1" -g   "((M+16-1)/16) * ((N+128-1)/128), 1, 1"  



python3.8  ${link_file}  ${wint8_dir}/*.h -o ${wint8_dir}/wint8

#python3.8 ${gen_paddle_file}  -on "triton_wint8" -kn "wint8_kernel" -kid "0,1,2" -header "wint8.h" -oi 2 -cof "generated/wint8/triton_win8.cu" -s "*fp16, *u8, *fp16, *fp16, *fp16, i32,i32,i32,  i32,i32,  i32,i32, i32,i32"







# mv the .c file to .cu file
for file in `find ./generated/ -name "*.c"`
do
    mv ${file} ${file}u
done


# 安装triton算子和运行单元测试

python3.8 setup_cuda.py install
#python3.8 matmul_test.py
CUDA_VISIBLE_DEVICES=4 python3.8 test_wint8.py 
