
# clone triton and generate
rm -rf generated
git clone https://github.com/openai/triton.git
cd triton
git reset --hard 2217bd2f5c271009f50ab2d2a639bbbb407a2650
cd -


# these two files are used to compile and link
compile_file=triton/python/triton/tools/compile.py
link_file=triton/python/triton/tools/link.py



matmul_dir=/zhoukangkang/2023-04-26SM80/Paddle/paddle/phi/kernels/fusion/custom_triton/generated/aot/matmul/fp16
mkdir -p ${matmul_dir}

python3.8  ${compile_file}     \
/zhoukangkang/triton/python/tutorials/03-matrix-multiplication-paddle.py     \
-n matmul_kernel   \
-o ${matmul_dir}/matmul_kernel_fp16     \
--out-name matmul_kernel_fp16     \
-w 8     -ns 3     \
-s "*fp16:16, *fp16:16, *fp16:16, i32:16,i32:16,i32:16, i32:16,i32:1,i32:16,i32:1,i32:16,i32:1, 128,256,64,8,2"     \
-g "(M+127)/128 * (N+255)/256, 1, 1"

python3.8  ${link_file}  ${matmul_dir}/*.h -o ${matmul_dir}/../matmul_fp16


# mv the .c file to .cu file
for file in `find ./generated/ -name "*.c"`
do
    mv ${file} ${file}u
done


# 安装triton算子和运行单元测试

# python3.8 setup_cuda.py install
# python3.8 test.py










