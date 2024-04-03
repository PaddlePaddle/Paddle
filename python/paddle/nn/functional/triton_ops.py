import paddle
from paddle.framework import in_dynamic_or_pir_mode

from paddle.base.layer_helper import LayerHelper

__all__ = []

import triton
import triton.language as tl

# @triton.autotune(
# 	configs=[
#         #triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1}, num_stages=4, num_warps=4),
#         #triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1}, num_stages=4, num_warps=4),
#         #triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1}, num_stages=4, num_warps=4),
#         #triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1}, num_stages=4, num_warps=4),
#         #triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1}, num_stages=4, num_warps=8),
#         triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1}, num_stages=4, num_warps=4),

# 		#triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
# 		#triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
# 		#triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
# 		#triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=8),
# 		#triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
# 		#triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4),
# 		#triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4),
#         #triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 16}, num_stages=2, num_warps=4),
#         #triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 512, 'GROUP_SIZE_M': 16}, num_stages=2, num_warps=4),
# 		# triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
# 		# triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
# 		# triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
# 		# triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=8),
# 		# triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
# 		# triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4),
# 		# triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4),
#         # triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 16}, num_stages=2, num_warps=4),
#         # triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 512, 'GROUP_SIZE_M': 16}, num_stages=2, num_warps=4),
#         # triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
# 		# triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
# 		# triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
# 		# triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=8),
# 		# triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
# 		# triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4),
# 		# triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4),
#         # triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 16}, num_stages=2, num_warps=4),
#         # triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 512, 'GROUP_SIZE_M': 16}, num_stages=2, num_warps=4),
# 		# triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
# 		# triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
# 		# triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
# 		# triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=8),
# 		# triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
# 		# triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4),
# 		# triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4),
#         # triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 16}, num_stages=2, num_warps=4),
#         # triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 512, 'GROUP_SIZE_M': 16}, num_stages=2, num_warps=4),

#  ],
# 	key=['M', 'N', 'K'],
#     reset_to_zero=['c_ptr']
# )
@triton.jit
def wint8_kernel(
    a_ptr, b_ptr, c_ptr,
    bs_ptr, bias_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, SPLIT_K: tl.constexpr
    ):
    """
    assert K % (BLOCK_SIZE_K * SPLIT_K) == 0
    """

    pid = tl.program_id(axis=0)
    pid_sp_k = tl.program_id(axis=1)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m    
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    #offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    
    offs_k = pid_sp_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    # 下面这个offs_k是传统的split-k
    # offs_k = pid_sp_k * (K // SPLIT_K) + tl.arange(0, BLOCK_SIZE_K)
    
    offs_k = tl.max_contiguous(tl.multiple_of(offs_k, BLOCK_SIZE_K), BLOCK_SIZE_K)
    

    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    magic_number = (0x00006400)
    magic_number = magic_number.to(tl.uint16)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)):
        
        #a = tl.load(a_ptrs, mask=offs_am[:, None] < M, other=0.0)
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)

        #fp_b = b.to(tl.float16)
        
        fp_b = b | magic_number
        fp_b = fp_b.to(tl.float16, bitcast=True)
        fp_b = fp_b - 1152

        bs_ptrs = bs_ptr + offs_bn[None, :]
        bs = tl.load(bs_ptrs)
        fp_b = fp_b * bs
        
        accumulator += tl.dot(a, fp_b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_bk
        
        #a_ptrs += BLOCK_SIZE_K * stride_ak
        #b_ptrs += BLOCK_SIZE_K * stride_bk
    
    # only let the first block do epilogue
    if bias_ptr is not None and pid_sp_k == 0:
        bias_ptrs = bias_ptr + offs_bn
        bias = tl.load(bias_ptrs)
        accumulator += bias[None,:]
    
    # bs_ptrs = bs_ptr + offs_bn[None, :]
    # bs = tl.load(bs_ptrs)
    # accumulator = (accumulator * bs)

    c = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    
    if SPLIT_K == 1:
        tl.store(c_ptrs, c, mask=c_mask)
    else:
        tl.atomic_add(c_ptrs, c, mask=c_mask)




triton_wint8_template = """
//#include <cuda_fp16.h>
#include <vector>
#include "${op_name}_kernel.h"
#include "paddle/extension.h"

std::map<std::vector<int>, int> map_problem_${op_name};

std::vector<paddle::Tensor> ${op_name}_func(
    const paddle::Tensor& x,
    const paddle::Tensor& qweight,
    const paddle::Tensor& scales,
    const paddle::Tensor& bias,
    bool bool_trans_w,
    bool with_bias) {
  int m = x.shape()[0];
  int k = x.shape()[1];
  int n = scales.shape()[0];

  auto c_out = paddle::full({m, n}, 0, x.dtype(), x.place());

  auto dev_x = x.data<phi::dtype::float16>();
  auto dev_weight = qweight.data<uint8_t>();
  auto dev_c = c_out.data<phi::dtype::float16>();
  auto dev_scales = scales.data<phi::dtype::float16>();
  const phi::dtype::float16* dev_bias = nullptr;
  if (with_bias) {
    dev_bias = bias.data<phi::dtype::float16>();
  }

  int stride_bk = n;
  int stride_bn = 1;

  if (bool_trans_w) {
    stride_bk = 1;
    stride_bn = k;
  }

  std::vector<int> problem_size = {m, k, n};

  if (map_problem_${op_name}.count(problem_size)) {
    int algo_id = map_problem_${op_name}[problem_size];
    auto status = ${op_name}_kernel(c_out.stream(),
                               (CUdeviceptr)(dev_x),
                               (CUdeviceptr)(dev_weight),
                               (CUdeviceptr)(dev_c),
                               (CUdeviceptr)(dev_scales),
                               (CUdeviceptr)(dev_bias),
                               m,
                               n,
                               k,
                               k,
                               1,
                               stride_bk,
                               stride_bn,
                               n,
                               1,
                               algo_id);
    assert(status == CUDA_SUCCESS);
    return {c_out};
  }

  float min_time = 10000.f;
  int select_id = -1;
  constexpr int WARMUP = 5;
  constexpr int REPEAT = 10;

  for (int algo_id = 0; algo_id < ${op_name}_kernel_get_num_algos(); ++algo_id) {
    cudaEvent_t beg[REPEAT];
    cudaEvent_t end[REPEAT];
    float elapsed_times[REPEAT];

    auto status = CUDA_SUCCESS;

    for (int ii = 0; ii < WARMUP + REPEAT; ii++) {
      int repeat_id = ii - WARMUP;

      if (repeat_id >= 0) {
        (cudaEventCreate(beg + repeat_id));
        (cudaEventCreate(end + repeat_id));
        (cudaEventRecord(beg[repeat_id]));
      }

      auto flush_l2_cache = paddle::full(
          {10 * 1024 * 1024}, 0, paddle::DataType::INT32, x.place());
      // std::cout << &flush_l2_cache  << std::endl;

      cudaMemset(dev_c, 0, sizeof(phi::dtype::float16) * m * n);
      status = ${op_name}_kernel(c_out.stream(),
                            (CUdeviceptr)(dev_x),
                            (CUdeviceptr)(dev_weight),
                            (CUdeviceptr)(dev_c),
                            (CUdeviceptr)(dev_scales),
                            (CUdeviceptr)(dev_bias),
                            m,
                            n,
                            k,
                            k,
                            1,
                            stride_bk,
                            stride_bn,
                            n,
                            1,
                            algo_id);
      // assert(status == CUDA_SUCCESS);

      if (repeat_id >= 0) {
        (cudaEventRecord(end[repeat_id]));
        (cudaEventSynchronize(end[repeat_id]));
        (cudaEventElapsedTime(
            elapsed_times + repeat_id, beg[repeat_id], end[repeat_id]));
      }
    }

    float avg_elapsed_time = 0.f;
    for (int ii = 0; ii < REPEAT; ++ii) {
      avg_elapsed_time += elapsed_times[ii];
    }

    if (avg_elapsed_time < min_time && status == CUDA_SUCCESS) {
      min_time = avg_elapsed_time;
      select_id = algo_id;
    }
  }

  map_problem_${op_name}[problem_size] = select_id;
  std::cout << "select algo id: " << select_id << std::endl;

  return {c_out};
}

std::vector<std::vector<int64_t>> ${op_name}_InferShape(const std::vector<int64_t>& a_shape,
                                                        const std::vector<int64_t>& b_shape) {
    return {{a_shape[0], b_shape[0]}};
}

std::vector<paddle::DataType> ${op_name}_InferDtype(const paddle::DataType& A_dtype) {
    return {A_dtype};
}

PD_BUILD_OP(${op_name})
    .Inputs({"x", "qweight", "scales", "bias"})
    .Outputs({"out"})
    .SetKernelFn(PD_KERNEL(${op_name}_func))
    .Attrs({"bool_trans_w: bool", "with_bias: bool"})
    .SetInferDtypeFn(PD_INFER_DTYPE(${op_name}_InferDtype))
    .SetInferShapeFn(PD_INFER_SHAPE(${op_name}_InferShape));
"""

template_install= """
import subprocess
a = subprocess.run(["find", "./", "-name", "*.cu"], stdout=subprocess.PIPE)
a = a.stdout.decode("utf-8").split("\\n")

generated_cu = []

for i in range(len(a)):
    print(a[i])
    if a[i] != "":
        generated_cu += [a[i]]

import paddle
from paddle.utils.cpp_extension import CUDAExtension, setup


def get_gencode_flags():
    prop = paddle.device.cuda.get_device_properties()
    cc = prop.major * 10 + prop.minor
    return ["-gencode", "arch=compute_{{0}},code=sm_{{0}}".format(cc)]


gencode_flags = get_gencode_flags()



setup(
    name="{package_name}",
    ext_modules=CUDAExtension(
        sources = generated_cu,
        extra_compile_args={{
            "cc": ["-lcuda"],
            "nvcc": [
                "-O3",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
            ]
            + gencode_flags,
        }},
        extra_link_args = ["-lcuda"]
    ),
)
"""

import re
def SubstituteTemplate(template, values):
    text = template
    changed = True
    while changed:
        changed = False
        for key, value in values.items():
            regex = "\\$\\{%s\\}" % key
            newtext = re.sub(regex, value, text)
            if newtext != text:
                changed = True
            text = newtext
    return text

def matmul_dequantize_int8_s2(x, 
                              qweight, 
                              scales, 
                              bias = None, 
                              bool_trans_w = True):
    """
    """

    print("x", x.shape)
    print("qweight", qweight)
    print("scales", scales.shape)

    if in_dynamic_or_pir_mode():
        #from triton_kernels import wint8_kernel
        M ,K = x.shape
        assert x.is_contiguous(), ""
        assert qweight.is_contiguous(), ""

        if bool_trans_w:
            N = qweight.shape[0]
            stride_bk = 1
            stride_bn = K
        else:
            N = qweight.shape[1]
            N = qweight.shape[0]
            stride_bk = N
            stride_bn = 1

        output = paddle.zeros((M, N), dtype=paddle.float16)
        
        grid = lambda META: (
            triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
            META['SPLIT_K'],
        )

        wint8_kernel[grid](
            x, qweight, output,
            scales,  bias,
            M, N, K,
            K, 1,  # A矩阵永远是行rowmajor
            stride_bk, stride_bn,
            N, 1,  # C矩阵永远是rowmajor
        )
        return output

    import os
    this_file = os.path.realpath(__file__)
    this_file = "/zhoukangkang/2023-04-26SM80/Paddle/python/paddle/nn/functional/A.py"
    
    op_name = "triton_wint8"
    if bias is not None:
        op_name += "_bias"
    if bool_trans_w:
        op_name += "_trans"
    address_hint = "*fp16:16, *u8:16, *fp16:16, *fp16:16, *fp16:16,"
    value_hint = "i32,i32:16,i32:16,  i32:16,i32:1,  i32:1,i32:16, i32:16,i32:1," 
    meta_para = "16, 128, 64, 1, 1"
    op_signature = address_hint + value_hint + meta_para
    generated_dir = "/root/.cache/haha/" + op_name
    os.makedirs(generated_dir, exist_ok=True)
    
    op_dict = {}
    op_dict["op_name"] = op_name
    file_path = generated_dir + "/" + op_name + ".cu"

    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            f.write(SubstituteTemplate(triton_wint8_template, op_dict))
            f.close()
        
        compile_file="/zhoukangkang/2023-04-26SM80/Paddle/paddle/phi/kernels/fusion/custom_triton/triton/python/triton/tools/compile.py"
        link_file="/zhoukangkang/2023-04-26SM80/Paddle/paddle/phi/kernels/fusion/custom_triton/triton/python/triton/tools/link.py"
        import sys
        python_path = sys.executable
        # AOT生成命令模板
        aot_template = '''
        {python_path}   {compile_file} {triton_py_file}   \
        -n wint8_kernel   \
        -o {generated_dir}/{op_name}_kernel     \
        --out-name {op_name}_kernel     \
        -w 4   -ns 2  \
        -s   "*fp16:16, *u8:16, *fp16:16, *fp16:16, *fp16:16,i32,i32:16,i32:16,  i32:16,i32:1,  i32:1,i32:16, i32:16,i32:1,16, 128, 64, 1, 1"   \
        -g   "((M+16-1)/16) * ((N+128-1)/128), 1, 1" \
        '''
        codegen_command = aot_template.format(python_path=python_path,compile_file=compile_file,triton_py_file=this_file,generated_dir=generated_dir,op_name=op_name)
        print("codegen_command", codegen_command)
        re = os.system(codegen_command)
        assert re == 0
        
        link_command = "{python_path}  {link_file}  {generated_dir}/*.h -o {generated_dir}/{op_name}_kernel".format(python_path=python_path, link_file=link_file, op_name=op_name, generated_dir=generated_dir)
        print("link_command", link_command)
        re = os.system(link_command)
        assert re == 0

        # 遍历目录中的文件,重命名
        for filename in os.listdir(generated_dir):
            if filename.endswith(".c"):
                old_path = os.path.join(generated_dir, filename)
                new_path = os.path.join(generated_dir, filename + "u")
                os.rename(old_path, new_path)
        
        file_path = generated_dir + "/setup_cuda.py"
        package_name = "triton_ops_haha"
        with open(file_path, "w") as f:
            f.write(template_install.format(package_name = package_name))
            f.close()
        install_command = "cd {generated_dir} && {python_path} setup_cuda.py install".format(generated_dir=generated_dir, python_path=python_path)
        print("install_command", install_command)
        re = os.system(install_command)
        assert re == 0

        #from triton_ops_hah import triton_wint8
        so_path = "/usr/local/lib/python3.8/dist-packages/triton_ops_hah-0.0.0-py3.8-linux-x86_64.egg/triton_ops_haha_pd_.so"
        paddle.utils.cpp_extension.load_op_meta_info_and_register_op(so_path)
    
    import triton_ops_haha

    helper = LayerHelper(
        op_name, **locals()
    )
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type=op_name,
        inputs={
            'x': x,
            'qweight': qweight,
            'scales': scales,
            'bias': bias,
        },
        attrs={
            "bool_trans_w": True,
            "with_bias": True,
        },
        outputs={'out': out},
    )
    return out

