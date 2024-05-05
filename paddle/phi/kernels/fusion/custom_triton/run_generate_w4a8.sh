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


w4a8_dir=generated/w4a8/
mkdir -p ${w4a8_dir}



python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 64, 64, 1, 1" -g   "((M+16-1)/16) * ((N+64-1)/64), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 64, 64, 1, 2" -g   "((M+16-1)/16) * ((N+64-1)/64), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 64, 64, 1, 4" -g   "((M+16-1)/16) * ((N+64-1)/64), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 64, 64, 1, 8" -g   "((M+16-1)/16) * ((N+64-1)/64), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 64, 128, 1, 1" -g   "((M+16-1)/16) * ((N+64-1)/64), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 64, 128, 1, 2" -g   "((M+16-1)/16) * ((N+64-1)/64), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 64, 128, 1, 4" -g   "((M+16-1)/16) * ((N+64-1)/64), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 64, 128, 1, 8" -g   "((M+16-1)/16) * ((N+64-1)/64), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 64, 256, 1, 1" -g   "((M+16-1)/16) * ((N+64-1)/64), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 64, 256, 1, 2" -g   "((M+16-1)/16) * ((N+64-1)/64), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 64, 256, 1, 4" -g   "((M+16-1)/16) * ((N+64-1)/64), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 64, 256, 1, 8" -g   "((M+16-1)/16) * ((N+64-1)/64), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 128, 64, 1, 1" -g   "((M+16-1)/16) * ((N+128-1)/128), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 128, 64, 1, 2" -g   "((M+16-1)/16) * ((N+128-1)/128), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 128, 64, 1, 4" -g   "((M+16-1)/16) * ((N+128-1)/128), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 128, 64, 1, 8" -g   "((M+16-1)/16) * ((N+128-1)/128), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 128, 128, 1, 1" -g   "((M+16-1)/16) * ((N+128-1)/128), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 128, 128, 1, 2" -g   "((M+16-1)/16) * ((N+128-1)/128), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 128, 128, 1, 4" -g   "((M+16-1)/16) * ((N+128-1)/128), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 128, 128, 1, 8" -g   "((M+16-1)/16) * ((N+128-1)/128), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 128, 256, 1, 1" -g   "((M+16-1)/16) * ((N+128-1)/128), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 128, 256, 1, 2" -g   "((M+16-1)/16) * ((N+128-1)/128), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 128, 256, 1, 4" -g   "((M+16-1)/16) * ((N+128-1)/128), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 128, 256, 1, 8" -g   "((M+16-1)/16) * ((N+128-1)/128), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 256, 64, 1, 1" -g   "((M+16-1)/16) * ((N+256-1)/256), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 256, 64, 1, 2" -g   "((M+16-1)/16) * ((N+256-1)/256), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 256, 64, 1, 4" -g   "((M+16-1)/16) * ((N+256-1)/256), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 256, 64, 1, 8" -g   "((M+16-1)/16) * ((N+256-1)/256), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 256, 128, 1, 1" -g   "((M+16-1)/16) * ((N+256-1)/256), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 256, 128, 1, 2" -g   "((M+16-1)/16) * ((N+256-1)/256), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 256, 128, 1, 4" -g   "((M+16-1)/16) * ((N+256-1)/256), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 256, 128, 1, 8" -g   "((M+16-1)/16) * ((N+256-1)/256), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 256, 256, 1, 1" -g   "((M+16-1)/16) * ((N+256-1)/256), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 256, 256, 1, 2" -g   "((M+16-1)/16) * ((N+256-1)/256), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 256, 256, 1, 4" -g   "((M+16-1)/16) * ((N+256-1)/256), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 256, 256, 1, 8" -g   "((M+16-1)/16) * ((N+256-1)/256), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 64, 64, 1, 1" -g   "((M+32-1)/32) * ((N+64-1)/64), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 64, 64, 1, 2" -g   "((M+32-1)/32) * ((N+64-1)/64), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 64, 64, 1, 4" -g   "((M+32-1)/32) * ((N+64-1)/64), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 64, 64, 1, 8" -g   "((M+32-1)/32) * ((N+64-1)/64), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 64, 128, 1, 1" -g   "((M+32-1)/32) * ((N+64-1)/64), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 64, 128, 1, 2" -g   "((M+32-1)/32) * ((N+64-1)/64), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 64, 128, 1, 4" -g   "((M+32-1)/32) * ((N+64-1)/64), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 64, 128, 1, 8" -g   "((M+32-1)/32) * ((N+64-1)/64), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 64, 256, 1, 1" -g   "((M+32-1)/32) * ((N+64-1)/64), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 64, 256, 1, 2" -g   "((M+32-1)/32) * ((N+64-1)/64), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 64, 256, 1, 4" -g   "((M+32-1)/32) * ((N+64-1)/64), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 64, 256, 1, 8" -g   "((M+32-1)/32) * ((N+64-1)/64), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 128, 64, 1, 1" -g   "((M+32-1)/32) * ((N+128-1)/128), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 128, 64, 1, 2" -g   "((M+32-1)/32) * ((N+128-1)/128), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 128, 64, 1, 4" -g   "((M+32-1)/32) * ((N+128-1)/128), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 128, 64, 1, 8" -g   "((M+32-1)/32) * ((N+128-1)/128), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 128, 128, 1, 1" -g   "((M+32-1)/32) * ((N+128-1)/128), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 128, 128, 1, 2" -g   "((M+32-1)/32) * ((N+128-1)/128), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 128, 128, 1, 4" -g   "((M+32-1)/32) * ((N+128-1)/128), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 128, 128, 1, 8" -g   "((M+32-1)/32) * ((N+128-1)/128), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 128, 256, 1, 1" -g   "((M+32-1)/32) * ((N+128-1)/128), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 128, 256, 1, 2" -g   "((M+32-1)/32) * ((N+128-1)/128), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 128, 256, 1, 4" -g   "((M+32-1)/32) * ((N+128-1)/128), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 128, 256, 1, 8" -g   "((M+32-1)/32) * ((N+128-1)/128), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 256, 64, 1, 1" -g   "((M+32-1)/32) * ((N+256-1)/256), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 256, 64, 1, 2" -g   "((M+32-1)/32) * ((N+256-1)/256), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 256, 64, 1, 4" -g   "((M+32-1)/32) * ((N+256-1)/256), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 256, 64, 1, 8" -g   "((M+32-1)/32) * ((N+256-1)/256), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 256, 128, 1, 1" -g   "((M+32-1)/32) * ((N+256-1)/256), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 256, 128, 1, 2" -g   "((M+32-1)/32) * ((N+256-1)/256), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 256, 128, 1, 4" -g   "((M+32-1)/32) * ((N+256-1)/256), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 256, 128, 1, 8" -g   "((M+32-1)/32) * ((N+256-1)/256), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 256, 256, 1, 1" -g   "((M+32-1)/32) * ((N+256-1)/256), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 256, 256, 1, 2" -g   "((M+32-1)/32) * ((N+256-1)/256), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 256, 256, 1, 4" -g   "((M+32-1)/32) * ((N+256-1)/256), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 256, 256, 1, 8" -g   "((M+32-1)/32) * ((N+256-1)/256), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 64, 64, 1, 1" -g   "((M+64-1)/64) * ((N+64-1)/64), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 64, 64, 1, 2" -g   "((M+64-1)/64) * ((N+64-1)/64), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 64, 64, 1, 4" -g   "((M+64-1)/64) * ((N+64-1)/64), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 64, 64, 1, 8" -g   "((M+64-1)/64) * ((N+64-1)/64), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 64, 128, 1, 1" -g   "((M+64-1)/64) * ((N+64-1)/64), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 64, 128, 1, 2" -g   "((M+64-1)/64) * ((N+64-1)/64), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 64, 128, 1, 4" -g   "((M+64-1)/64) * ((N+64-1)/64), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 64, 128, 1, 8" -g   "((M+64-1)/64) * ((N+64-1)/64), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 64, 256, 1, 1" -g   "((M+64-1)/64) * ((N+64-1)/64), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 64, 256, 1, 2" -g   "((M+64-1)/64) * ((N+64-1)/64), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 64, 256, 1, 4" -g   "((M+64-1)/64) * ((N+64-1)/64), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 64, 256, 1, 8" -g   "((M+64-1)/64) * ((N+64-1)/64), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 128, 64, 1, 1" -g   "((M+64-1)/64) * ((N+128-1)/128), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 128, 64, 1, 2" -g   "((M+64-1)/64) * ((N+128-1)/128), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 128, 64, 1, 4" -g   "((M+64-1)/64) * ((N+128-1)/128), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 128, 64, 1, 8" -g   "((M+64-1)/64) * ((N+128-1)/128), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 128, 128, 1, 1" -g   "((M+64-1)/64) * ((N+128-1)/128), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 128, 128, 1, 2" -g   "((M+64-1)/64) * ((N+128-1)/128), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 128, 128, 1, 4" -g   "((M+64-1)/64) * ((N+128-1)/128), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 128, 128, 1, 8" -g   "((M+64-1)/64) * ((N+128-1)/128), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 128, 256, 1, 1" -g   "((M+64-1)/64) * ((N+128-1)/128), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 128, 256, 1, 2" -g   "((M+64-1)/64) * ((N+128-1)/128), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 128, 256, 1, 4" -g   "((M+64-1)/64) * ((N+128-1)/128), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 128, 256, 1, 8" -g   "((M+64-1)/64) * ((N+128-1)/128), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 256, 64, 1, 1" -g   "((M+64-1)/64) * ((N+256-1)/256), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 256, 64, 1, 2" -g   "((M+64-1)/64) * ((N+256-1)/256), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 256, 64, 1, 4" -g   "((M+64-1)/64) * ((N+256-1)/256), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 256, 64, 1, 8" -g   "((M+64-1)/64) * ((N+256-1)/256), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 256, 128, 1, 1" -g   "((M+64-1)/64) * ((N+256-1)/256), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 256, 128, 1, 2" -g   "((M+64-1)/64) * ((N+256-1)/256), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 256, 128, 1, 4" -g   "((M+64-1)/64) * ((N+256-1)/256), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 256, 128, 1, 8" -g   "((M+64-1)/64) * ((N+256-1)/256), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 256, 256, 1, 1" -g   "((M+64-1)/64) * ((N+256-1)/256), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 256, 256, 1, 2" -g   "((M+64-1)/64) * ((N+256-1)/256), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 256, 256, 1, 4" -g   "((M+64-1)/64) * ((N+256-1)/256), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 256, 256, 1, 8" -g   "((M+64-1)/64) * ((N+256-1)/256), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 64, 64, 1, 1" -g   "((M+128-1)/128) * ((N+64-1)/64), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 64, 64, 1, 2" -g   "((M+128-1)/128) * ((N+64-1)/64), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 64, 64, 1, 4" -g   "((M+128-1)/128) * ((N+64-1)/64), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 64, 64, 1, 8" -g   "((M+128-1)/128) * ((N+64-1)/64), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 64, 128, 1, 1" -g   "((M+128-1)/128) * ((N+64-1)/64), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 64, 128, 1, 2" -g   "((M+128-1)/128) * ((N+64-1)/64), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 64, 128, 1, 4" -g   "((M+128-1)/128) * ((N+64-1)/64), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 64, 128, 1, 8" -g   "((M+128-1)/128) * ((N+64-1)/64), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 64, 256, 1, 1" -g   "((M+128-1)/128) * ((N+64-1)/64), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 64, 256, 1, 2" -g   "((M+128-1)/128) * ((N+64-1)/64), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 64, 256, 1, 4" -g   "((M+128-1)/128) * ((N+64-1)/64), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 64, 256, 1, 8" -g   "((M+128-1)/128) * ((N+64-1)/64), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 128, 64, 1, 1" -g   "((M+128-1)/128) * ((N+128-1)/128), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 128, 64, 1, 2" -g   "((M+128-1)/128) * ((N+128-1)/128), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 128, 64, 1, 4" -g   "((M+128-1)/128) * ((N+128-1)/128), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 128, 64, 1, 8" -g   "((M+128-1)/128) * ((N+128-1)/128), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 128, 128, 1, 1" -g   "((M+128-1)/128) * ((N+128-1)/128), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 128, 128, 1, 2" -g   "((M+128-1)/128) * ((N+128-1)/128), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 128, 128, 1, 4" -g   "((M+128-1)/128) * ((N+128-1)/128), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 128, 128, 1, 8" -g   "((M+128-1)/128) * ((N+128-1)/128), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 128, 256, 1, 1" -g   "((M+128-1)/128) * ((N+128-1)/128), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 128, 256, 1, 2" -g   "((M+128-1)/128) * ((N+128-1)/128), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 128, 256, 1, 4" -g   "((M+128-1)/128) * ((N+128-1)/128), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 128, 256, 1, 8" -g   "((M+128-1)/128) * ((N+128-1)/128), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 8   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 256, 64, 1, 1" -g   "((M+128-1)/128) * ((N+256-1)/256), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 8   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 256, 64, 1, 2" -g   "((M+128-1)/128) * ((N+256-1)/256), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 8   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 256, 64, 1, 4" -g   "((M+128-1)/128) * ((N+256-1)/256), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 8   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 256, 64, 1, 8" -g   "((M+128-1)/128) * ((N+256-1)/256), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 8   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 256, 128, 1, 1" -g   "((M+128-1)/128) * ((N+256-1)/256), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 8   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 256, 128, 1, 2" -g   "((M+128-1)/128) * ((N+256-1)/256), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 8   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 256, 128, 1, 4" -g   "((M+128-1)/128) * ((N+256-1)/256), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 8   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 256, 128, 1, 8" -g   "((M+128-1)/128) * ((N+256-1)/256), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 8   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 256, 256, 1, 1" -g   "((M+128-1)/128) * ((N+256-1)/256), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 8   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 256, 256, 1, 2" -g   "((M+128-1)/128) * ((N+256-1)/256), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 8   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 256, 256, 1, 4" -g   "((M+128-1)/128) * ((N+256-1)/256), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 8   -ns 2 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 256, 256, 1, 8" -g   "((M+128-1)/128) * ((N+256-1)/256), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 64, 64, 1, 1" -g   "((M+16-1)/16) * ((N+64-1)/64), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 64, 64, 1, 2" -g   "((M+16-1)/16) * ((N+64-1)/64), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 64, 64, 1, 4" -g   "((M+16-1)/16) * ((N+64-1)/64), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 64, 64, 1, 8" -g   "((M+16-1)/16) * ((N+64-1)/64), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 64, 128, 1, 1" -g   "((M+16-1)/16) * ((N+64-1)/64), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 64, 128, 1, 2" -g   "((M+16-1)/16) * ((N+64-1)/64), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 64, 128, 1, 4" -g   "((M+16-1)/16) * ((N+64-1)/64), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 64, 128, 1, 8" -g   "((M+16-1)/16) * ((N+64-1)/64), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 64, 256, 1, 1" -g   "((M+16-1)/16) * ((N+64-1)/64), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 64, 256, 1, 2" -g   "((M+16-1)/16) * ((N+64-1)/64), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 64, 256, 1, 4" -g   "((M+16-1)/16) * ((N+64-1)/64), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 64, 256, 1, 8" -g   "((M+16-1)/16) * ((N+64-1)/64), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 128, 64, 1, 1" -g   "((M+16-1)/16) * ((N+128-1)/128), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 128, 64, 1, 2" -g   "((M+16-1)/16) * ((N+128-1)/128), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 128, 64, 1, 4" -g   "((M+16-1)/16) * ((N+128-1)/128), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 128, 64, 1, 8" -g   "((M+16-1)/16) * ((N+128-1)/128), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 128, 128, 1, 1" -g   "((M+16-1)/16) * ((N+128-1)/128), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 128, 128, 1, 2" -g   "((M+16-1)/16) * ((N+128-1)/128), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 128, 128, 1, 4" -g   "((M+16-1)/16) * ((N+128-1)/128), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 128, 128, 1, 8" -g   "((M+16-1)/16) * ((N+128-1)/128), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 128, 256, 1, 1" -g   "((M+16-1)/16) * ((N+128-1)/128), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 128, 256, 1, 2" -g   "((M+16-1)/16) * ((N+128-1)/128), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 128, 256, 1, 4" -g   "((M+16-1)/16) * ((N+128-1)/128), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 128, 256, 1, 8" -g   "((M+16-1)/16) * ((N+128-1)/128), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 256, 64, 1, 1" -g   "((M+16-1)/16) * ((N+256-1)/256), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 256, 64, 1, 2" -g   "((M+16-1)/16) * ((N+256-1)/256), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 256, 64, 1, 4" -g   "((M+16-1)/16) * ((N+256-1)/256), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 256, 64, 1, 8" -g   "((M+16-1)/16) * ((N+256-1)/256), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 256, 128, 1, 1" -g   "((M+16-1)/16) * ((N+256-1)/256), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 256, 128, 1, 2" -g   "((M+16-1)/16) * ((N+256-1)/256), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 256, 128, 1, 4" -g   "((M+16-1)/16) * ((N+256-1)/256), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 256, 128, 1, 8" -g   "((M+16-1)/16) * ((N+256-1)/256), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 256, 256, 1, 1" -g   "((M+16-1)/16) * ((N+256-1)/256), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 256, 256, 1, 2" -g   "((M+16-1)/16) * ((N+256-1)/256), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 256, 256, 1, 4" -g   "((M+16-1)/16) * ((N+256-1)/256), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 256, 256, 1, 8" -g   "((M+16-1)/16) * ((N+256-1)/256), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 64, 64, 1, 1" -g   "((M+32-1)/32) * ((N+64-1)/64), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 64, 64, 1, 2" -g   "((M+32-1)/32) * ((N+64-1)/64), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 64, 64, 1, 4" -g   "((M+32-1)/32) * ((N+64-1)/64), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 64, 64, 1, 8" -g   "((M+32-1)/32) * ((N+64-1)/64), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 64, 128, 1, 1" -g   "((M+32-1)/32) * ((N+64-1)/64), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 64, 128, 1, 2" -g   "((M+32-1)/32) * ((N+64-1)/64), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 64, 128, 1, 4" -g   "((M+32-1)/32) * ((N+64-1)/64), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 64, 128, 1, 8" -g   "((M+32-1)/32) * ((N+64-1)/64), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 64, 256, 1, 1" -g   "((M+32-1)/32) * ((N+64-1)/64), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 64, 256, 1, 2" -g   "((M+32-1)/32) * ((N+64-1)/64), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 64, 256, 1, 4" -g   "((M+32-1)/32) * ((N+64-1)/64), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 64, 256, 1, 8" -g   "((M+32-1)/32) * ((N+64-1)/64), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 128, 64, 1, 1" -g   "((M+32-1)/32) * ((N+128-1)/128), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 128, 64, 1, 2" -g   "((M+32-1)/32) * ((N+128-1)/128), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 128, 64, 1, 4" -g   "((M+32-1)/32) * ((N+128-1)/128), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 128, 64, 1, 8" -g   "((M+32-1)/32) * ((N+128-1)/128), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 128, 128, 1, 1" -g   "((M+32-1)/32) * ((N+128-1)/128), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 128, 128, 1, 2" -g   "((M+32-1)/32) * ((N+128-1)/128), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 128, 128, 1, 4" -g   "((M+32-1)/32) * ((N+128-1)/128), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 128, 128, 1, 8" -g   "((M+32-1)/32) * ((N+128-1)/128), 8, 1"  &
wait

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 128, 256, 1, 1" -g   "((M+32-1)/32) * ((N+128-1)/128), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 128, 256, 1, 2" -g   "((M+32-1)/32) * ((N+128-1)/128), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 128, 256, 1, 4" -g   "((M+32-1)/32) * ((N+128-1)/128), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 128, 256, 1, 8" -g   "((M+32-1)/32) * ((N+128-1)/128), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 256, 64, 1, 1" -g   "((M+32-1)/32) * ((N+256-1)/256), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 256, 64, 1, 2" -g   "((M+32-1)/32) * ((N+256-1)/256), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 256, 64, 1, 4" -g   "((M+32-1)/32) * ((N+256-1)/256), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 256, 64, 1, 8" -g   "((M+32-1)/32) * ((N+256-1)/256), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 256, 128, 1, 1" -g   "((M+32-1)/32) * ((N+256-1)/256), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 256, 128, 1, 2" -g   "((M+32-1)/32) * ((N+256-1)/256), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 256, 128, 1, 4" -g   "((M+32-1)/32) * ((N+256-1)/256), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 256, 128, 1, 8" -g   "((M+32-1)/32) * ((N+256-1)/256), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 256, 256, 1, 1" -g   "((M+32-1)/32) * ((N+256-1)/256), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 256, 256, 1, 2" -g   "((M+32-1)/32) * ((N+256-1)/256), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 256, 256, 1, 4" -g   "((M+32-1)/32) * ((N+256-1)/256), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 256, 256, 1, 8" -g   "((M+32-1)/32) * ((N+256-1)/256), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 64, 64, 1, 1" -g   "((M+64-1)/64) * ((N+64-1)/64), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 64, 64, 1, 2" -g   "((M+64-1)/64) * ((N+64-1)/64), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 64, 64, 1, 4" -g   "((M+64-1)/64) * ((N+64-1)/64), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 64, 64, 1, 8" -g   "((M+64-1)/64) * ((N+64-1)/64), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 64, 128, 1, 1" -g   "((M+64-1)/64) * ((N+64-1)/64), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 64, 128, 1, 2" -g   "((M+64-1)/64) * ((N+64-1)/64), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 64, 128, 1, 4" -g   "((M+64-1)/64) * ((N+64-1)/64), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 64, 128, 1, 8" -g   "((M+64-1)/64) * ((N+64-1)/64), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 64, 256, 1, 1" -g   "((M+64-1)/64) * ((N+64-1)/64), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 64, 256, 1, 2" -g   "((M+64-1)/64) * ((N+64-1)/64), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 64, 256, 1, 4" -g   "((M+64-1)/64) * ((N+64-1)/64), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 64, 256, 1, 8" -g   "((M+64-1)/64) * ((N+64-1)/64), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 128, 64, 1, 1" -g   "((M+64-1)/64) * ((N+128-1)/128), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 128, 64, 1, 2" -g   "((M+64-1)/64) * ((N+128-1)/128), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 128, 64, 1, 4" -g   "((M+64-1)/64) * ((N+128-1)/128), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 128, 64, 1, 8" -g   "((M+64-1)/64) * ((N+128-1)/128), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 128, 128, 1, 1" -g   "((M+64-1)/64) * ((N+128-1)/128), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 128, 128, 1, 2" -g   "((M+64-1)/64) * ((N+128-1)/128), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 128, 128, 1, 4" -g   "((M+64-1)/64) * ((N+128-1)/128), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 128, 128, 1, 8" -g   "((M+64-1)/64) * ((N+128-1)/128), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 128, 256, 1, 1" -g   "((M+64-1)/64) * ((N+128-1)/128), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 128, 256, 1, 2" -g   "((M+64-1)/64) * ((N+128-1)/128), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 128, 256, 1, 4" -g   "((M+64-1)/64) * ((N+128-1)/128), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 128, 256, 1, 8" -g   "((M+64-1)/64) * ((N+128-1)/128), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 256, 64, 1, 1" -g   "((M+64-1)/64) * ((N+256-1)/256), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 256, 64, 1, 2" -g   "((M+64-1)/64) * ((N+256-1)/256), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 256, 64, 1, 4" -g   "((M+64-1)/64) * ((N+256-1)/256), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 256, 64, 1, 8" -g   "((M+64-1)/64) * ((N+256-1)/256), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 256, 128, 1, 1" -g   "((M+64-1)/64) * ((N+256-1)/256), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 256, 128, 1, 2" -g   "((M+64-1)/64) * ((N+256-1)/256), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 256, 128, 1, 4" -g   "((M+64-1)/64) * ((N+256-1)/256), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 256, 128, 1, 8" -g   "((M+64-1)/64) * ((N+256-1)/256), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 256, 256, 1, 1" -g   "((M+64-1)/64) * ((N+256-1)/256), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 256, 256, 1, 2" -g   "((M+64-1)/64) * ((N+256-1)/256), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 256, 256, 1, 4" -g   "((M+64-1)/64) * ((N+256-1)/256), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 256, 256, 1, 8" -g   "((M+64-1)/64) * ((N+256-1)/256), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 64, 64, 1, 1" -g   "((M+128-1)/128) * ((N+64-1)/64), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 64, 64, 1, 2" -g   "((M+128-1)/128) * ((N+64-1)/64), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 64, 64, 1, 4" -g   "((M+128-1)/128) * ((N+64-1)/64), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 64, 64, 1, 8" -g   "((M+128-1)/128) * ((N+64-1)/64), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 64, 128, 1, 1" -g   "((M+128-1)/128) * ((N+64-1)/64), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 64, 128, 1, 2" -g   "((M+128-1)/128) * ((N+64-1)/64), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 64, 128, 1, 4" -g   "((M+128-1)/128) * ((N+64-1)/64), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 64, 128, 1, 8" -g   "((M+128-1)/128) * ((N+64-1)/64), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 64, 256, 1, 1" -g   "((M+128-1)/128) * ((N+64-1)/64), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 64, 256, 1, 2" -g   "((M+128-1)/128) * ((N+64-1)/64), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 64, 256, 1, 4" -g   "((M+128-1)/128) * ((N+64-1)/64), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 64, 256, 1, 8" -g   "((M+128-1)/128) * ((N+64-1)/64), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 128, 64, 1, 1" -g   "((M+128-1)/128) * ((N+128-1)/128), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 128, 64, 1, 2" -g   "((M+128-1)/128) * ((N+128-1)/128), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 128, 64, 1, 4" -g   "((M+128-1)/128) * ((N+128-1)/128), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 128, 64, 1, 8" -g   "((M+128-1)/128) * ((N+128-1)/128), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 128, 128, 1, 1" -g   "((M+128-1)/128) * ((N+128-1)/128), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 128, 128, 1, 2" -g   "((M+128-1)/128) * ((N+128-1)/128), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 128, 128, 1, 4" -g   "((M+128-1)/128) * ((N+128-1)/128), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 128, 128, 1, 8" -g   "((M+128-1)/128) * ((N+128-1)/128), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 128, 256, 1, 1" -g   "((M+128-1)/128) * ((N+128-1)/128), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 128, 256, 1, 2" -g   "((M+128-1)/128) * ((N+128-1)/128), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 128, 256, 1, 4" -g   "((M+128-1)/128) * ((N+128-1)/128), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 128, 256, 1, 8" -g   "((M+128-1)/128) * ((N+128-1)/128), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 8   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 256, 64, 1, 1" -g   "((M+128-1)/128) * ((N+256-1)/256), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 8   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 256, 64, 1, 2" -g   "((M+128-1)/128) * ((N+256-1)/256), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 8   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 256, 64, 1, 4" -g   "((M+128-1)/128) * ((N+256-1)/256), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 8   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 256, 64, 1, 8" -g   "((M+128-1)/128) * ((N+256-1)/256), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 8   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 256, 128, 1, 1" -g   "((M+128-1)/128) * ((N+256-1)/256), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 8   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 256, 128, 1, 2" -g   "((M+128-1)/128) * ((N+256-1)/256), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 8   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 256, 128, 1, 4" -g   "((M+128-1)/128) * ((N+256-1)/256), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 8   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 256, 128, 1, 8" -g   "((M+128-1)/128) * ((N+256-1)/256), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 8   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 256, 256, 1, 1" -g   "((M+128-1)/128) * ((N+256-1)/256), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 8   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 256, 256, 1, 2" -g   "((M+128-1)/128) * ((N+256-1)/256), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 8   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 256, 256, 1, 4" -g   "((M+128-1)/128) * ((N+256-1)/256), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 8   -ns 3 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 256, 256, 1, 8" -g   "((M+128-1)/128) * ((N+256-1)/256), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 64, 64, 1, 1" -g   "((M+16-1)/16) * ((N+64-1)/64), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 64, 64, 1, 2" -g   "((M+16-1)/16) * ((N+64-1)/64), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 64, 64, 1, 4" -g   "((M+16-1)/16) * ((N+64-1)/64), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 64, 64, 1, 8" -g   "((M+16-1)/16) * ((N+64-1)/64), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 64, 128, 1, 1" -g   "((M+16-1)/16) * ((N+64-1)/64), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 64, 128, 1, 2" -g   "((M+16-1)/16) * ((N+64-1)/64), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 64, 128, 1, 4" -g   "((M+16-1)/16) * ((N+64-1)/64), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 64, 128, 1, 8" -g   "((M+16-1)/16) * ((N+64-1)/64), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 64, 256, 1, 1" -g   "((M+16-1)/16) * ((N+64-1)/64), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 64, 256, 1, 2" -g   "((M+16-1)/16) * ((N+64-1)/64), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 64, 256, 1, 4" -g   "((M+16-1)/16) * ((N+64-1)/64), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 64, 256, 1, 8" -g   "((M+16-1)/16) * ((N+64-1)/64), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 128, 64, 1, 1" -g   "((M+16-1)/16) * ((N+128-1)/128), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 128, 64, 1, 2" -g   "((M+16-1)/16) * ((N+128-1)/128), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 128, 64, 1, 4" -g   "((M+16-1)/16) * ((N+128-1)/128), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 128, 64, 1, 8" -g   "((M+16-1)/16) * ((N+128-1)/128), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 128, 128, 1, 1" -g   "((M+16-1)/16) * ((N+128-1)/128), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 128, 128, 1, 2" -g   "((M+16-1)/16) * ((N+128-1)/128), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 128, 128, 1, 4" -g   "((M+16-1)/16) * ((N+128-1)/128), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 128, 128, 1, 8" -g   "((M+16-1)/16) * ((N+128-1)/128), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 128, 256, 1, 1" -g   "((M+16-1)/16) * ((N+128-1)/128), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 128, 256, 1, 2" -g   "((M+16-1)/16) * ((N+128-1)/128), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 128, 256, 1, 4" -g   "((M+16-1)/16) * ((N+128-1)/128), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 128, 256, 1, 8" -g   "((M+16-1)/16) * ((N+128-1)/128), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 256, 64, 1, 1" -g   "((M+16-1)/16) * ((N+256-1)/256), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 256, 64, 1, 2" -g   "((M+16-1)/16) * ((N+256-1)/256), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 256, 64, 1, 4" -g   "((M+16-1)/16) * ((N+256-1)/256), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 256, 64, 1, 8" -g   "((M+16-1)/16) * ((N+256-1)/256), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 256, 128, 1, 1" -g   "((M+16-1)/16) * ((N+256-1)/256), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 256, 128, 1, 2" -g   "((M+16-1)/16) * ((N+256-1)/256), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 256, 128, 1, 4" -g   "((M+16-1)/16) * ((N+256-1)/256), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 256, 128, 1, 8" -g   "((M+16-1)/16) * ((N+256-1)/256), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 256, 256, 1, 1" -g   "((M+16-1)/16) * ((N+256-1)/256), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 256, 256, 1, 2" -g   "((M+16-1)/16) * ((N+256-1)/256), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 256, 256, 1, 4" -g   "((M+16-1)/16) * ((N+256-1)/256), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 256, 256, 1, 8" -g   "((M+16-1)/16) * ((N+256-1)/256), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 64, 64, 1, 1" -g   "((M+32-1)/32) * ((N+64-1)/64), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 64, 64, 1, 2" -g   "((M+32-1)/32) * ((N+64-1)/64), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 64, 64, 1, 4" -g   "((M+32-1)/32) * ((N+64-1)/64), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 64, 64, 1, 8" -g   "((M+32-1)/32) * ((N+64-1)/64), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 64, 128, 1, 1" -g   "((M+32-1)/32) * ((N+64-1)/64), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 64, 128, 1, 2" -g   "((M+32-1)/32) * ((N+64-1)/64), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 64, 128, 1, 4" -g   "((M+32-1)/32) * ((N+64-1)/64), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 64, 128, 1, 8" -g   "((M+32-1)/32) * ((N+64-1)/64), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 64, 256, 1, 1" -g   "((M+32-1)/32) * ((N+64-1)/64), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 64, 256, 1, 2" -g   "((M+32-1)/32) * ((N+64-1)/64), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 64, 256, 1, 4" -g   "((M+32-1)/32) * ((N+64-1)/64), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 64, 256, 1, 8" -g   "((M+32-1)/32) * ((N+64-1)/64), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 128, 64, 1, 1" -g   "((M+32-1)/32) * ((N+128-1)/128), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 128, 64, 1, 2" -g   "((M+32-1)/32) * ((N+128-1)/128), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 128, 64, 1, 4" -g   "((M+32-1)/32) * ((N+128-1)/128), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 128, 64, 1, 8" -g   "((M+32-1)/32) * ((N+128-1)/128), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 128, 128, 1, 1" -g   "((M+32-1)/32) * ((N+128-1)/128), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 128, 128, 1, 2" -g   "((M+32-1)/32) * ((N+128-1)/128), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 128, 128, 1, 4" -g   "((M+32-1)/32) * ((N+128-1)/128), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 128, 128, 1, 8" -g   "((M+32-1)/32) * ((N+128-1)/128), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 128, 256, 1, 1" -g   "((M+32-1)/32) * ((N+128-1)/128), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 128, 256, 1, 2" -g   "((M+32-1)/32) * ((N+128-1)/128), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 128, 256, 1, 4" -g   "((M+32-1)/32) * ((N+128-1)/128), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 128, 256, 1, 8" -g   "((M+32-1)/32) * ((N+128-1)/128), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 256, 64, 1, 1" -g   "((M+32-1)/32) * ((N+256-1)/256), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 256, 64, 1, 2" -g   "((M+32-1)/32) * ((N+256-1)/256), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 256, 64, 1, 4" -g   "((M+32-1)/32) * ((N+256-1)/256), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 256, 64, 1, 8" -g   "((M+32-1)/32) * ((N+256-1)/256), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 256, 128, 1, 1" -g   "((M+32-1)/32) * ((N+256-1)/256), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 256, 128, 1, 2" -g   "((M+32-1)/32) * ((N+256-1)/256), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 256, 128, 1, 4" -g   "((M+32-1)/32) * ((N+256-1)/256), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 256, 128, 1, 8" -g   "((M+32-1)/32) * ((N+256-1)/256), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 256, 256, 1, 1" -g   "((M+32-1)/32) * ((N+256-1)/256), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 256, 256, 1, 2" -g   "((M+32-1)/32) * ((N+256-1)/256), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 256, 256, 1, 4" -g   "((M+32-1)/32) * ((N+256-1)/256), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 256, 256, 1, 8" -g   "((M+32-1)/32) * ((N+256-1)/256), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 64, 64, 1, 1" -g   "((M+64-1)/64) * ((N+64-1)/64), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 64, 64, 1, 2" -g   "((M+64-1)/64) * ((N+64-1)/64), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 64, 64, 1, 4" -g   "((M+64-1)/64) * ((N+64-1)/64), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 64, 64, 1, 8" -g   "((M+64-1)/64) * ((N+64-1)/64), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 64, 128, 1, 1" -g   "((M+64-1)/64) * ((N+64-1)/64), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 64, 128, 1, 2" -g   "((M+64-1)/64) * ((N+64-1)/64), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 64, 128, 1, 4" -g   "((M+64-1)/64) * ((N+64-1)/64), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 64, 128, 1, 8" -g   "((M+64-1)/64) * ((N+64-1)/64), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 64, 256, 1, 1" -g   "((M+64-1)/64) * ((N+64-1)/64), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 64, 256, 1, 2" -g   "((M+64-1)/64) * ((N+64-1)/64), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 64, 256, 1, 4" -g   "((M+64-1)/64) * ((N+64-1)/64), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 64, 256, 1, 8" -g   "((M+64-1)/64) * ((N+64-1)/64), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 128, 64, 1, 1" -g   "((M+64-1)/64) * ((N+128-1)/128), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 128, 64, 1, 2" -g   "((M+64-1)/64) * ((N+128-1)/128), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 128, 64, 1, 4" -g   "((M+64-1)/64) * ((N+128-1)/128), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 128, 64, 1, 8" -g   "((M+64-1)/64) * ((N+128-1)/128), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 128, 128, 1, 1" -g   "((M+64-1)/64) * ((N+128-1)/128), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 128, 128, 1, 2" -g   "((M+64-1)/64) * ((N+128-1)/128), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 128, 128, 1, 4" -g   "((M+64-1)/64) * ((N+128-1)/128), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 128, 128, 1, 8" -g   "((M+64-1)/64) * ((N+128-1)/128), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 128, 256, 1, 1" -g   "((M+64-1)/64) * ((N+128-1)/128), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 128, 256, 1, 2" -g   "((M+64-1)/64) * ((N+128-1)/128), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 128, 256, 1, 4" -g   "((M+64-1)/64) * ((N+128-1)/128), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 128, 256, 1, 8" -g   "((M+64-1)/64) * ((N+128-1)/128), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 256, 64, 1, 1" -g   "((M+64-1)/64) * ((N+256-1)/256), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 256, 64, 1, 2" -g   "((M+64-1)/64) * ((N+256-1)/256), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 256, 64, 1, 4" -g   "((M+64-1)/64) * ((N+256-1)/256), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 256, 64, 1, 8" -g   "((M+64-1)/64) * ((N+256-1)/256), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 256, 128, 1, 1" -g   "((M+64-1)/64) * ((N+256-1)/256), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 256, 128, 1, 2" -g   "((M+64-1)/64) * ((N+256-1)/256), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 256, 128, 1, 4" -g   "((M+64-1)/64) * ((N+256-1)/256), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 256, 128, 1, 8" -g   "((M+64-1)/64) * ((N+256-1)/256), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 256, 256, 1, 1" -g   "((M+64-1)/64) * ((N+256-1)/256), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 256, 256, 1, 2" -g   "((M+64-1)/64) * ((N+256-1)/256), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 256, 256, 1, 4" -g   "((M+64-1)/64) * ((N+256-1)/256), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 256, 256, 1, 8" -g   "((M+64-1)/64) * ((N+256-1)/256), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 64, 64, 1, 1" -g   "((M+128-1)/128) * ((N+64-1)/64), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 64, 64, 1, 2" -g   "((M+128-1)/128) * ((N+64-1)/64), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 64, 64, 1, 4" -g   "((M+128-1)/128) * ((N+64-1)/64), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 64, 64, 1, 8" -g   "((M+128-1)/128) * ((N+64-1)/64), 8, 1"  &
wait

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 64, 128, 1, 1" -g   "((M+128-1)/128) * ((N+64-1)/64), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 64, 128, 1, 2" -g   "((M+128-1)/128) * ((N+64-1)/64), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 64, 128, 1, 4" -g   "((M+128-1)/128) * ((N+64-1)/64), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 64, 128, 1, 8" -g   "((M+128-1)/128) * ((N+64-1)/64), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 64, 256, 1, 1" -g   "((M+128-1)/128) * ((N+64-1)/64), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 64, 256, 1, 2" -g   "((M+128-1)/128) * ((N+64-1)/64), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 64, 256, 1, 4" -g   "((M+128-1)/128) * ((N+64-1)/64), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 64, 256, 1, 8" -g   "((M+128-1)/128) * ((N+64-1)/64), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 128, 64, 1, 1" -g   "((M+128-1)/128) * ((N+128-1)/128), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 128, 64, 1, 2" -g   "((M+128-1)/128) * ((N+128-1)/128), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 128, 64, 1, 4" -g   "((M+128-1)/128) * ((N+128-1)/128), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 128, 64, 1, 8" -g   "((M+128-1)/128) * ((N+128-1)/128), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 128, 128, 1, 1" -g   "((M+128-1)/128) * ((N+128-1)/128), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 128, 128, 1, 2" -g   "((M+128-1)/128) * ((N+128-1)/128), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 128, 128, 1, 4" -g   "((M+128-1)/128) * ((N+128-1)/128), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 128, 128, 1, 8" -g   "((M+128-1)/128) * ((N+128-1)/128), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 128, 256, 1, 1" -g   "((M+128-1)/128) * ((N+128-1)/128), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 128, 256, 1, 2" -g   "((M+128-1)/128) * ((N+128-1)/128), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 128, 256, 1, 4" -g   "((M+128-1)/128) * ((N+128-1)/128), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 128, 256, 1, 8" -g   "((M+128-1)/128) * ((N+128-1)/128), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 8   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 256, 64, 1, 1" -g   "((M+128-1)/128) * ((N+256-1)/256), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 8   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 256, 64, 1, 2" -g   "((M+128-1)/128) * ((N+256-1)/256), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 8   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 256, 64, 1, 4" -g   "((M+128-1)/128) * ((N+256-1)/256), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 8   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 256, 64, 1, 8" -g   "((M+128-1)/128) * ((N+256-1)/256), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 8   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 256, 128, 1, 1" -g   "((M+128-1)/128) * ((N+256-1)/256), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 8   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 256, 128, 1, 2" -g   "((M+128-1)/128) * ((N+256-1)/256), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 8   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 256, 128, 1, 4" -g   "((M+128-1)/128) * ((N+256-1)/256), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 8   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 256, 128, 1, 8" -g   "((M+128-1)/128) * ((N+256-1)/256), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 8   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 256, 256, 1, 1" -g   "((M+128-1)/128) * ((N+256-1)/256), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 8   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 256, 256, 1, 2" -g   "((M+128-1)/128) * ((N+256-1)/256), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 8   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 256, 256, 1, 4" -g   "((M+128-1)/128) * ((N+256-1)/256), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 8   -ns 4 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 256, 256, 1, 8" -g   "((M+128-1)/128) * ((N+256-1)/256), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 64, 64, 1, 1" -g   "((M+16-1)/16) * ((N+64-1)/64), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 64, 64, 1, 2" -g   "((M+16-1)/16) * ((N+64-1)/64), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 64, 64, 1, 4" -g   "((M+16-1)/16) * ((N+64-1)/64), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 64, 64, 1, 8" -g   "((M+16-1)/16) * ((N+64-1)/64), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 64, 128, 1, 1" -g   "((M+16-1)/16) * ((N+64-1)/64), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 64, 128, 1, 2" -g   "((M+16-1)/16) * ((N+64-1)/64), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 64, 128, 1, 4" -g   "((M+16-1)/16) * ((N+64-1)/64), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 64, 128, 1, 8" -g   "((M+16-1)/16) * ((N+64-1)/64), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 64, 256, 1, 1" -g   "((M+16-1)/16) * ((N+64-1)/64), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 64, 256, 1, 2" -g   "((M+16-1)/16) * ((N+64-1)/64), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 64, 256, 1, 4" -g   "((M+16-1)/16) * ((N+64-1)/64), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 64, 256, 1, 8" -g   "((M+16-1)/16) * ((N+64-1)/64), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 128, 64, 1, 1" -g   "((M+16-1)/16) * ((N+128-1)/128), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 128, 64, 1, 2" -g   "((M+16-1)/16) * ((N+128-1)/128), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 128, 64, 1, 4" -g   "((M+16-1)/16) * ((N+128-1)/128), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 128, 64, 1, 8" -g   "((M+16-1)/16) * ((N+128-1)/128), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 128, 128, 1, 1" -g   "((M+16-1)/16) * ((N+128-1)/128), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 128, 128, 1, 2" -g   "((M+16-1)/16) * ((N+128-1)/128), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 128, 128, 1, 4" -g   "((M+16-1)/16) * ((N+128-1)/128), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 128, 128, 1, 8" -g   "((M+16-1)/16) * ((N+128-1)/128), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 128, 256, 1, 1" -g   "((M+16-1)/16) * ((N+128-1)/128), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 128, 256, 1, 2" -g   "((M+16-1)/16) * ((N+128-1)/128), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 128, 256, 1, 4" -g   "((M+16-1)/16) * ((N+128-1)/128), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 128, 256, 1, 8" -g   "((M+16-1)/16) * ((N+128-1)/128), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 256, 64, 1, 1" -g   "((M+16-1)/16) * ((N+256-1)/256), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 256, 64, 1, 2" -g   "((M+16-1)/16) * ((N+256-1)/256), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 256, 64, 1, 4" -g   "((M+16-1)/16) * ((N+256-1)/256), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 256, 64, 1, 8" -g   "((M+16-1)/16) * ((N+256-1)/256), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 256, 128, 1, 1" -g   "((M+16-1)/16) * ((N+256-1)/256), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 256, 128, 1, 2" -g   "((M+16-1)/16) * ((N+256-1)/256), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 256, 128, 1, 4" -g   "((M+16-1)/16) * ((N+256-1)/256), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 256, 128, 1, 8" -g   "((M+16-1)/16) * ((N+256-1)/256), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 256, 256, 1, 1" -g   "((M+16-1)/16) * ((N+256-1)/256), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 256, 256, 1, 2" -g   "((M+16-1)/16) * ((N+256-1)/256), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 256, 256, 1, 4" -g   "((M+16-1)/16) * ((N+256-1)/256), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 256, 256, 1, 8" -g   "((M+16-1)/16) * ((N+256-1)/256), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 64, 64, 1, 1" -g   "((M+32-1)/32) * ((N+64-1)/64), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 64, 64, 1, 2" -g   "((M+32-1)/32) * ((N+64-1)/64), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 64, 64, 1, 4" -g   "((M+32-1)/32) * ((N+64-1)/64), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 64, 64, 1, 8" -g   "((M+32-1)/32) * ((N+64-1)/64), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 64, 128, 1, 1" -g   "((M+32-1)/32) * ((N+64-1)/64), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 64, 128, 1, 2" -g   "((M+32-1)/32) * ((N+64-1)/64), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 64, 128, 1, 4" -g   "((M+32-1)/32) * ((N+64-1)/64), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 64, 128, 1, 8" -g   "((M+32-1)/32) * ((N+64-1)/64), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 64, 256, 1, 1" -g   "((M+32-1)/32) * ((N+64-1)/64), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 64, 256, 1, 2" -g   "((M+32-1)/32) * ((N+64-1)/64), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 64, 256, 1, 4" -g   "((M+32-1)/32) * ((N+64-1)/64), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 64, 256, 1, 8" -g   "((M+32-1)/32) * ((N+64-1)/64), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 128, 64, 1, 1" -g   "((M+32-1)/32) * ((N+128-1)/128), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 128, 64, 1, 2" -g   "((M+32-1)/32) * ((N+128-1)/128), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 128, 64, 1, 4" -g   "((M+32-1)/32) * ((N+128-1)/128), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 128, 64, 1, 8" -g   "((M+32-1)/32) * ((N+128-1)/128), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 128, 128, 1, 1" -g   "((M+32-1)/32) * ((N+128-1)/128), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 128, 128, 1, 2" -g   "((M+32-1)/32) * ((N+128-1)/128), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 128, 128, 1, 4" -g   "((M+32-1)/32) * ((N+128-1)/128), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 128, 128, 1, 8" -g   "((M+32-1)/32) * ((N+128-1)/128), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 128, 256, 1, 1" -g   "((M+32-1)/32) * ((N+128-1)/128), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 128, 256, 1, 2" -g   "((M+32-1)/32) * ((N+128-1)/128), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 128, 256, 1, 4" -g   "((M+32-1)/32) * ((N+128-1)/128), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 128, 256, 1, 8" -g   "((M+32-1)/32) * ((N+128-1)/128), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 256, 64, 1, 1" -g   "((M+32-1)/32) * ((N+256-1)/256), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 256, 64, 1, 2" -g   "((M+32-1)/32) * ((N+256-1)/256), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 256, 64, 1, 4" -g   "((M+32-1)/32) * ((N+256-1)/256), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 256, 64, 1, 8" -g   "((M+32-1)/32) * ((N+256-1)/256), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 256, 128, 1, 1" -g   "((M+32-1)/32) * ((N+256-1)/256), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 256, 128, 1, 2" -g   "((M+32-1)/32) * ((N+256-1)/256), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 256, 128, 1, 4" -g   "((M+32-1)/32) * ((N+256-1)/256), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 256, 128, 1, 8" -g   "((M+32-1)/32) * ((N+256-1)/256), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 256, 256, 1, 1" -g   "((M+32-1)/32) * ((N+256-1)/256), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 256, 256, 1, 2" -g   "((M+32-1)/32) * ((N+256-1)/256), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 256, 256, 1, 4" -g   "((M+32-1)/32) * ((N+256-1)/256), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 256, 256, 1, 8" -g   "((M+32-1)/32) * ((N+256-1)/256), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 64, 64, 1, 1" -g   "((M+64-1)/64) * ((N+64-1)/64), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 64, 64, 1, 2" -g   "((M+64-1)/64) * ((N+64-1)/64), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 64, 64, 1, 4" -g   "((M+64-1)/64) * ((N+64-1)/64), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 64, 64, 1, 8" -g   "((M+64-1)/64) * ((N+64-1)/64), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 64, 128, 1, 1" -g   "((M+64-1)/64) * ((N+64-1)/64), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 64, 128, 1, 2" -g   "((M+64-1)/64) * ((N+64-1)/64), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 64, 128, 1, 4" -g   "((M+64-1)/64) * ((N+64-1)/64), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 64, 128, 1, 8" -g   "((M+64-1)/64) * ((N+64-1)/64), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 64, 256, 1, 1" -g   "((M+64-1)/64) * ((N+64-1)/64), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 64, 256, 1, 2" -g   "((M+64-1)/64) * ((N+64-1)/64), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 64, 256, 1, 4" -g   "((M+64-1)/64) * ((N+64-1)/64), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 64, 256, 1, 8" -g   "((M+64-1)/64) * ((N+64-1)/64), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 128, 64, 1, 1" -g   "((M+64-1)/64) * ((N+128-1)/128), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 128, 64, 1, 2" -g   "((M+64-1)/64) * ((N+128-1)/128), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 128, 64, 1, 4" -g   "((M+64-1)/64) * ((N+128-1)/128), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 128, 64, 1, 8" -g   "((M+64-1)/64) * ((N+128-1)/128), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 128, 128, 1, 1" -g   "((M+64-1)/64) * ((N+128-1)/128), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 128, 128, 1, 2" -g   "((M+64-1)/64) * ((N+128-1)/128), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 128, 128, 1, 4" -g   "((M+64-1)/64) * ((N+128-1)/128), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 128, 128, 1, 8" -g   "((M+64-1)/64) * ((N+128-1)/128), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 128, 256, 1, 1" -g   "((M+64-1)/64) * ((N+128-1)/128), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 128, 256, 1, 2" -g   "((M+64-1)/64) * ((N+128-1)/128), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 128, 256, 1, 4" -g   "((M+64-1)/64) * ((N+128-1)/128), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 128, 256, 1, 8" -g   "((M+64-1)/64) * ((N+128-1)/128), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 256, 64, 1, 1" -g   "((M+64-1)/64) * ((N+256-1)/256), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 256, 64, 1, 2" -g   "((M+64-1)/64) * ((N+256-1)/256), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 256, 64, 1, 4" -g   "((M+64-1)/64) * ((N+256-1)/256), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 256, 64, 1, 8" -g   "((M+64-1)/64) * ((N+256-1)/256), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 256, 128, 1, 1" -g   "((M+64-1)/64) * ((N+256-1)/256), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 256, 128, 1, 2" -g   "((M+64-1)/64) * ((N+256-1)/256), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 256, 128, 1, 4" -g   "((M+64-1)/64) * ((N+256-1)/256), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 256, 128, 1, 8" -g   "((M+64-1)/64) * ((N+256-1)/256), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 256, 256, 1, 1" -g   "((M+64-1)/64) * ((N+256-1)/256), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 256, 256, 1, 2" -g   "((M+64-1)/64) * ((N+256-1)/256), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 256, 256, 1, 4" -g   "((M+64-1)/64) * ((N+256-1)/256), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 256, 256, 1, 8" -g   "((M+64-1)/64) * ((N+256-1)/256), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 64, 64, 1, 1" -g   "((M+128-1)/128) * ((N+64-1)/64), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 64, 64, 1, 2" -g   "((M+128-1)/128) * ((N+64-1)/64), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 64, 64, 1, 4" -g   "((M+128-1)/128) * ((N+64-1)/64), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 64, 64, 1, 8" -g   "((M+128-1)/128) * ((N+64-1)/64), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 64, 128, 1, 1" -g   "((M+128-1)/128) * ((N+64-1)/64), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 64, 128, 1, 2" -g   "((M+128-1)/128) * ((N+64-1)/64), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 64, 128, 1, 4" -g   "((M+128-1)/128) * ((N+64-1)/64), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 64, 128, 1, 8" -g   "((M+128-1)/128) * ((N+64-1)/64), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 64, 256, 1, 1" -g   "((M+128-1)/128) * ((N+64-1)/64), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 64, 256, 1, 2" -g   "((M+128-1)/128) * ((N+64-1)/64), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 64, 256, 1, 4" -g   "((M+128-1)/128) * ((N+64-1)/64), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 64, 256, 1, 8" -g   "((M+128-1)/128) * ((N+64-1)/64), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 128, 64, 1, 1" -g   "((M+128-1)/128) * ((N+128-1)/128), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 128, 64, 1, 2" -g   "((M+128-1)/128) * ((N+128-1)/128), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 128, 64, 1, 4" -g   "((M+128-1)/128) * ((N+128-1)/128), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 128, 64, 1, 8" -g   "((M+128-1)/128) * ((N+128-1)/128), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 128, 128, 1, 1" -g   "((M+128-1)/128) * ((N+128-1)/128), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 128, 128, 1, 2" -g   "((M+128-1)/128) * ((N+128-1)/128), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 128, 128, 1, 4" -g   "((M+128-1)/128) * ((N+128-1)/128), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 128, 128, 1, 8" -g   "((M+128-1)/128) * ((N+128-1)/128), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 128, 256, 1, 1" -g   "((M+128-1)/128) * ((N+128-1)/128), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 128, 256, 1, 2" -g   "((M+128-1)/128) * ((N+128-1)/128), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 128, 256, 1, 4" -g   "((M+128-1)/128) * ((N+128-1)/128), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 128, 256, 1, 8" -g   "((M+128-1)/128) * ((N+128-1)/128), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 8   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 256, 64, 1, 1" -g   "((M+128-1)/128) * ((N+256-1)/256), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 8   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 256, 64, 1, 2" -g   "((M+128-1)/128) * ((N+256-1)/256), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 8   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 256, 64, 1, 4" -g   "((M+128-1)/128) * ((N+256-1)/256), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 8   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 256, 64, 1, 8" -g   "((M+128-1)/128) * ((N+256-1)/256), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 8   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 256, 128, 1, 1" -g   "((M+128-1)/128) * ((N+256-1)/256), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 8   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 256, 128, 1, 2" -g   "((M+128-1)/128) * ((N+256-1)/256), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 8   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 256, 128, 1, 4" -g   "((M+128-1)/128) * ((N+256-1)/256), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 8   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 256, 128, 1, 8" -g   "((M+128-1)/128) * ((N+256-1)/256), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 8   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 256, 256, 1, 1" -g   "((M+128-1)/128) * ((N+256-1)/256), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 8   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 256, 256, 1, 2" -g   "((M+128-1)/128) * ((N+256-1)/256), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 8   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 256, 256, 1, 4" -g   "((M+128-1)/128) * ((N+256-1)/256), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 8   -ns 5 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 256, 256, 1, 8" -g   "((M+128-1)/128) * ((N+256-1)/256), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 64, 64, 1, 1" -g   "((M+16-1)/16) * ((N+64-1)/64), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 64, 64, 1, 2" -g   "((M+16-1)/16) * ((N+64-1)/64), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 64, 64, 1, 4" -g   "((M+16-1)/16) * ((N+64-1)/64), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 64, 64, 1, 8" -g   "((M+16-1)/16) * ((N+64-1)/64), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 64, 128, 1, 1" -g   "((M+16-1)/16) * ((N+64-1)/64), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 64, 128, 1, 2" -g   "((M+16-1)/16) * ((N+64-1)/64), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 64, 128, 1, 4" -g   "((M+16-1)/16) * ((N+64-1)/64), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 64, 128, 1, 8" -g   "((M+16-1)/16) * ((N+64-1)/64), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 64, 256, 1, 1" -g   "((M+16-1)/16) * ((N+64-1)/64), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 64, 256, 1, 2" -g   "((M+16-1)/16) * ((N+64-1)/64), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 64, 256, 1, 4" -g   "((M+16-1)/16) * ((N+64-1)/64), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 64, 256, 1, 8" -g   "((M+16-1)/16) * ((N+64-1)/64), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 128, 64, 1, 1" -g   "((M+16-1)/16) * ((N+128-1)/128), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 128, 64, 1, 2" -g   "((M+16-1)/16) * ((N+128-1)/128), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 128, 64, 1, 4" -g   "((M+16-1)/16) * ((N+128-1)/128), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 128, 64, 1, 8" -g   "((M+16-1)/16) * ((N+128-1)/128), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 128, 128, 1, 1" -g   "((M+16-1)/16) * ((N+128-1)/128), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 128, 128, 1, 2" -g   "((M+16-1)/16) * ((N+128-1)/128), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 128, 128, 1, 4" -g   "((M+16-1)/16) * ((N+128-1)/128), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 128, 128, 1, 8" -g   "((M+16-1)/16) * ((N+128-1)/128), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 128, 256, 1, 1" -g   "((M+16-1)/16) * ((N+128-1)/128), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 128, 256, 1, 2" -g   "((M+16-1)/16) * ((N+128-1)/128), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 128, 256, 1, 4" -g   "((M+16-1)/16) * ((N+128-1)/128), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 128, 256, 1, 8" -g   "((M+16-1)/16) * ((N+128-1)/128), 8, 1"  &
wait

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 256, 64, 1, 1" -g   "((M+16-1)/16) * ((N+256-1)/256), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 256, 64, 1, 2" -g   "((M+16-1)/16) * ((N+256-1)/256), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 256, 64, 1, 4" -g   "((M+16-1)/16) * ((N+256-1)/256), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 256, 64, 1, 8" -g   "((M+16-1)/16) * ((N+256-1)/256), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 256, 128, 1, 1" -g   "((M+16-1)/16) * ((N+256-1)/256), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 256, 128, 1, 2" -g   "((M+16-1)/16) * ((N+256-1)/256), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 256, 128, 1, 4" -g   "((M+16-1)/16) * ((N+256-1)/256), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 256, 128, 1, 8" -g   "((M+16-1)/16) * ((N+256-1)/256), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 256, 256, 1, 1" -g   "((M+16-1)/16) * ((N+256-1)/256), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 256, 256, 1, 2" -g   "((M+16-1)/16) * ((N+256-1)/256), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 256, 256, 1, 4" -g   "((M+16-1)/16) * ((N+256-1)/256), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 16, 256, 256, 1, 8" -g   "((M+16-1)/16) * ((N+256-1)/256), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 64, 64, 1, 1" -g   "((M+32-1)/32) * ((N+64-1)/64), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 64, 64, 1, 2" -g   "((M+32-1)/32) * ((N+64-1)/64), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 64, 64, 1, 4" -g   "((M+32-1)/32) * ((N+64-1)/64), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 64, 64, 1, 8" -g   "((M+32-1)/32) * ((N+64-1)/64), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 64, 128, 1, 1" -g   "((M+32-1)/32) * ((N+64-1)/64), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 64, 128, 1, 2" -g   "((M+32-1)/32) * ((N+64-1)/64), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 64, 128, 1, 4" -g   "((M+32-1)/32) * ((N+64-1)/64), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 64, 128, 1, 8" -g   "((M+32-1)/32) * ((N+64-1)/64), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 64, 256, 1, 1" -g   "((M+32-1)/32) * ((N+64-1)/64), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 64, 256, 1, 2" -g   "((M+32-1)/32) * ((N+64-1)/64), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 64, 256, 1, 4" -g   "((M+32-1)/32) * ((N+64-1)/64), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 64, 256, 1, 8" -g   "((M+32-1)/32) * ((N+64-1)/64), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 128, 64, 1, 1" -g   "((M+32-1)/32) * ((N+128-1)/128), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 128, 64, 1, 2" -g   "((M+32-1)/32) * ((N+128-1)/128), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 128, 64, 1, 4" -g   "((M+32-1)/32) * ((N+128-1)/128), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 128, 64, 1, 8" -g   "((M+32-1)/32) * ((N+128-1)/128), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 128, 128, 1, 1" -g   "((M+32-1)/32) * ((N+128-1)/128), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 128, 128, 1, 2" -g   "((M+32-1)/32) * ((N+128-1)/128), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 128, 128, 1, 4" -g   "((M+32-1)/32) * ((N+128-1)/128), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 128, 128, 1, 8" -g   "((M+32-1)/32) * ((N+128-1)/128), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 128, 256, 1, 1" -g   "((M+32-1)/32) * ((N+128-1)/128), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 128, 256, 1, 2" -g   "((M+32-1)/32) * ((N+128-1)/128), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 128, 256, 1, 4" -g   "((M+32-1)/32) * ((N+128-1)/128), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 128, 256, 1, 8" -g   "((M+32-1)/32) * ((N+128-1)/128), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 256, 64, 1, 1" -g   "((M+32-1)/32) * ((N+256-1)/256), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 256, 64, 1, 2" -g   "((M+32-1)/32) * ((N+256-1)/256), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 256, 64, 1, 4" -g   "((M+32-1)/32) * ((N+256-1)/256), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 256, 64, 1, 8" -g   "((M+32-1)/32) * ((N+256-1)/256), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 256, 128, 1, 1" -g   "((M+32-1)/32) * ((N+256-1)/256), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 256, 128, 1, 2" -g   "((M+32-1)/32) * ((N+256-1)/256), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 256, 128, 1, 4" -g   "((M+32-1)/32) * ((N+256-1)/256), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 256, 128, 1, 8" -g   "((M+32-1)/32) * ((N+256-1)/256), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 256, 256, 1, 1" -g   "((M+32-1)/32) * ((N+256-1)/256), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 256, 256, 1, 2" -g   "((M+32-1)/32) * ((N+256-1)/256), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 256, 256, 1, 4" -g   "((M+32-1)/32) * ((N+256-1)/256), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 32, 256, 256, 1, 8" -g   "((M+32-1)/32) * ((N+256-1)/256), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 64, 64, 1, 1" -g   "((M+64-1)/64) * ((N+64-1)/64), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 64, 64, 1, 2" -g   "((M+64-1)/64) * ((N+64-1)/64), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 64, 64, 1, 4" -g   "((M+64-1)/64) * ((N+64-1)/64), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 64, 64, 1, 8" -g   "((M+64-1)/64) * ((N+64-1)/64), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 64, 128, 1, 1" -g   "((M+64-1)/64) * ((N+64-1)/64), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 64, 128, 1, 2" -g   "((M+64-1)/64) * ((N+64-1)/64), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 64, 128, 1, 4" -g   "((M+64-1)/64) * ((N+64-1)/64), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 64, 128, 1, 8" -g   "((M+64-1)/64) * ((N+64-1)/64), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 64, 256, 1, 1" -g   "((M+64-1)/64) * ((N+64-1)/64), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 64, 256, 1, 2" -g   "((M+64-1)/64) * ((N+64-1)/64), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 64, 256, 1, 4" -g   "((M+64-1)/64) * ((N+64-1)/64), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 64, 256, 1, 8" -g   "((M+64-1)/64) * ((N+64-1)/64), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 128, 64, 1, 1" -g   "((M+64-1)/64) * ((N+128-1)/128), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 128, 64, 1, 2" -g   "((M+64-1)/64) * ((N+128-1)/128), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 128, 64, 1, 4" -g   "((M+64-1)/64) * ((N+128-1)/128), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 128, 64, 1, 8" -g   "((M+64-1)/64) * ((N+128-1)/128), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 128, 128, 1, 1" -g   "((M+64-1)/64) * ((N+128-1)/128), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 128, 128, 1, 2" -g   "((M+64-1)/64) * ((N+128-1)/128), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 128, 128, 1, 4" -g   "((M+64-1)/64) * ((N+128-1)/128), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 128, 128, 1, 8" -g   "((M+64-1)/64) * ((N+128-1)/128), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 128, 256, 1, 1" -g   "((M+64-1)/64) * ((N+128-1)/128), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 128, 256, 1, 2" -g   "((M+64-1)/64) * ((N+128-1)/128), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 128, 256, 1, 4" -g   "((M+64-1)/64) * ((N+128-1)/128), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 128, 256, 1, 8" -g   "((M+64-1)/64) * ((N+128-1)/128), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 256, 64, 1, 1" -g   "((M+64-1)/64) * ((N+256-1)/256), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 256, 64, 1, 2" -g   "((M+64-1)/64) * ((N+256-1)/256), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 256, 64, 1, 4" -g   "((M+64-1)/64) * ((N+256-1)/256), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 256, 64, 1, 8" -g   "((M+64-1)/64) * ((N+256-1)/256), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 256, 128, 1, 1" -g   "((M+64-1)/64) * ((N+256-1)/256), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 256, 128, 1, 2" -g   "((M+64-1)/64) * ((N+256-1)/256), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 256, 128, 1, 4" -g   "((M+64-1)/64) * ((N+256-1)/256), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 256, 128, 1, 8" -g   "((M+64-1)/64) * ((N+256-1)/256), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 256, 256, 1, 1" -g   "((M+64-1)/64) * ((N+256-1)/256), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 256, 256, 1, 2" -g   "((M+64-1)/64) * ((N+256-1)/256), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 256, 256, 1, 4" -g   "((M+64-1)/64) * ((N+256-1)/256), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 64, 256, 256, 1, 8" -g   "((M+64-1)/64) * ((N+256-1)/256), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 64, 64, 1, 1" -g   "((M+128-1)/128) * ((N+64-1)/64), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 64, 64, 1, 2" -g   "((M+128-1)/128) * ((N+64-1)/64), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 64, 64, 1, 4" -g   "((M+128-1)/128) * ((N+64-1)/64), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 64, 64, 1, 8" -g   "((M+128-1)/128) * ((N+64-1)/64), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 64, 128, 1, 1" -g   "((M+128-1)/128) * ((N+64-1)/64), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 64, 128, 1, 2" -g   "((M+128-1)/128) * ((N+64-1)/64), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 64, 128, 1, 4" -g   "((M+128-1)/128) * ((N+64-1)/64), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 64, 128, 1, 8" -g   "((M+128-1)/128) * ((N+64-1)/64), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 64, 256, 1, 1" -g   "((M+128-1)/128) * ((N+64-1)/64), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 64, 256, 1, 2" -g   "((M+128-1)/128) * ((N+64-1)/64), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 64, 256, 1, 4" -g   "((M+128-1)/128) * ((N+64-1)/64), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 64, 256, 1, 8" -g   "((M+128-1)/128) * ((N+64-1)/64), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 128, 64, 1, 1" -g   "((M+128-1)/128) * ((N+128-1)/128), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 128, 64, 1, 2" -g   "((M+128-1)/128) * ((N+128-1)/128), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 128, 64, 1, 4" -g   "((M+128-1)/128) * ((N+128-1)/128), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 128, 64, 1, 8" -g   "((M+128-1)/128) * ((N+128-1)/128), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 128, 128, 1, 1" -g   "((M+128-1)/128) * ((N+128-1)/128), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 128, 128, 1, 2" -g   "((M+128-1)/128) * ((N+128-1)/128), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 128, 128, 1, 4" -g   "((M+128-1)/128) * ((N+128-1)/128), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 128, 128, 1, 8" -g   "((M+128-1)/128) * ((N+128-1)/128), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 128, 256, 1, 1" -g   "((M+128-1)/128) * ((N+128-1)/128), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 128, 256, 1, 2" -g   "((M+128-1)/128) * ((N+128-1)/128), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 128, 256, 1, 4" -g   "((M+128-1)/128) * ((N+128-1)/128), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 4   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 128, 256, 1, 8" -g   "((M+128-1)/128) * ((N+128-1)/128), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 8   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 256, 64, 1, 1" -g   "((M+128-1)/128) * ((N+256-1)/256), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 8   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 256, 64, 1, 2" -g   "((M+128-1)/128) * ((N+256-1)/256), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 8   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 256, 64, 1, 4" -g   "((M+128-1)/128) * ((N+256-1)/256), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 8   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 256, 64, 1, 8" -g   "((M+128-1)/128) * ((N+256-1)/256), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 8   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 256, 128, 1, 1" -g   "((M+128-1)/128) * ((N+256-1)/256), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 8   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 256, 128, 1, 2" -g   "((M+128-1)/128) * ((N+256-1)/256), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 8   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 256, 128, 1, 4" -g   "((M+128-1)/128) * ((N+256-1)/256), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 8   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 256, 128, 1, 8" -g   "((M+128-1)/128) * ((N+256-1)/256), 8, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 8   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 256, 256, 1, 1" -g   "((M+128-1)/128) * ((N+256-1)/256), 1, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 8   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 256, 256, 1, 2" -g   "((M+128-1)/128) * ((N+256-1)/256), 2, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 8   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 256, 256, 1, 4" -g   "((M+128-1)/128) * ((N+256-1)/256), 4, 1"  &

python3.8  ${compile_file}     /zhoukangkang/triton/python/paddle_tutorials/w4a8.py    -n w4a8_kernel   -o ${w4a8_dir}/w4a8     --out-name w4a8_kernel     -w 8   -ns 6 -s   "*i8:16, *i32:16, *i32:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:16,i32:1, i32:16,i32:1, 128, 256, 256, 1, 8" -g   "((M+128-1)/128) * ((N+256-1)/256), 8, 1"  &
wait



python3.8  ${link_file}  ${w4a8_dir}/*.h -o ${w4a8_dir}/w4a8


# mv the .c file to .cu file
for file in `find ./generated/ -name "*.c"`
do
    mv ${file} ${file}u
done


# 安装triton算子和运行单元测试

python3.8 setup_cuda.py install
CUDA_VISIBLE_DEVICES=2 python3.8 test_w4a8.py 


