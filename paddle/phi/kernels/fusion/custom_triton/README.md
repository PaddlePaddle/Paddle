






# 如何添加一个Triton算子


## 编写Triton算子

用Python文件写好一个triton算子，借助于triton的aot工具，可以编译出cubin kernel。

python3.8  triton/python/triton/tools/compile.py     \
/zhoukangkang/triton/python/tutorials/03-matrix-multiplication-paddle.py     \
-n matmul_kernel   \
-o aot/fp16/matmul_kernel_fp16     \
--out-name matmul_kernel_fp16     \
-w 4     -ns 2     \
-s "*fp16:16, *fp16:16, *fp16:16,i32,i32,i32,i32,i32,i32,i32,i32,i32,128,256,64,8,2"     \
-g "(M+127)/128 * (N+255)/256, 1, 1"




python3.8 triton/python/triton/tools/link.py aot/fp16/*.h -o aot/matmul_kernel_fp16


