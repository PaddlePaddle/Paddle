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

import os

import triton
import triton.language as tl

import paddle
from paddle import _C_ops
from paddle.base.framework import OpProtoHolder
from paddle.base.layer_helper import LayerHelper
from paddle.framework import in_dynamic_or_pir_mode

from .triton_utils import (
    SubstituteTemplate,
    build_package,
    compile_file,
    extract_triton_kernel,
    find_so_path,
    get_op_name_with_suffix,
    get_pointer_hint,
    get_value_hint,
    link_file,
    multi_process_do,
    paddle_custom_op_head_part,
    python_path,
    rename_c_to_cu,
    tune_and_invoke_part,
)


def get_wint8_kernel_config():
    configs = []
    for num_stages in [2, 3, 4, 5, 6]:
        for block_m in [16, 32, 64, 128]:
            for block_n in [64, 128, 256]:
                for block_k in [64, 128, 256]:
                    for split_k in [1, 2, 4, 8]:
                        num_warps = 4
                        if block_m * block_n >= 128 * 256:
                            num_warps = 8

                        configs.append(
                            triton.Config(
                                {
                                    "SPLIT_K": split_k,
                                    "BLOCK_SIZE_M": block_m,
                                    "BLOCK_SIZE_N": block_n,
                                    "BLOCK_SIZE_K": block_k,
                                    "GROUP_SIZE_M": 1,
                                },
                                num_stages=num_stages,
                                num_warps=num_warps,
                            )
                        )
    return configs


@triton.autotune(
    configs=get_wint8_kernel_config(),
    key=['M', 'N', 'K'],
    reset_to_zero=['c_ptr'],
)
@triton.jit
def wint8_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    bs_ptr,
    bias_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
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

    # col major mapping
    # pid_m = pid // num_pid_n
    # pid_n = pid % num_pid_n

    # row major mapping
    # pid_m = pid % num_pid_m
    # pid_n = pid // num_pid_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    # offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N

    offs_k = pid_sp_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

    offs_k = tl.max_contiguous(
        tl.multiple_of(offs_k, BLOCK_SIZE_K), BLOCK_SIZE_K
    )

    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    magic_number = 0x00006400
    magic_number = magic_number.to(tl.uint16)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)):
        # a = tl.load(a_ptrs, mask=offs_am[:, None] < M, other=0.0)
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)

        # fp_b = b.to(tl.float16)

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

    # only let the first block do epilogue
    if bias_ptr is not None and pid_sp_k == 0:
        bias_ptrs = bias_ptr + offs_bn
        bias = tl.load(bias_ptrs)
        accumulator += bias[None, :]

    c = accumulator.to(tl.float16)

    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    if SPLIT_K == 1:
        tl.store(c_ptrs, c, mask=c_mask)
    else:
        tl.atomic_add(c_ptrs, c, mask=c_mask)


triton_wint8_template = (
    paddle_custom_op_head_part
    + """
std::vector<paddle::Tensor> ${op_name}_func(
    const paddle::Tensor& x,
    const paddle::Tensor& qweight,
    const paddle::Tensor& scales,
    paddle::optional<paddle::Tensor>& bias,
    bool bool_trans_w) {
  int m = x.shape()[0];
  int k = x.shape()[1];
  int n = scales.shape()[0];

  auto c_out = paddle::full({m, n}, 0, x.dtype(), x.place());

  auto dev_x = get_tensor_ptr(x);
  auto dev_weight = get_tensor_ptr(qweight);
  auto dev_c = get_tensor_ptr(c_out);
  auto dev_scales = get_tensor_ptr(scales);
  CUdeviceptr dev_bias = (CUdeviceptr)(nullptr);
  if (bias) {
    dev_bias = get_tensor_ptr(*bias);
  }

  int stride_bk = n;
  int stride_bn = 1;

  if (bool_trans_w) {
    stride_bk = 1;
    stride_bn = k;
  }

auto run_triton_kernel = [&](int algo_id) -> CUresult{
    return ${op_name}_kernel(c_out.stream(),
                               dev_x,
                               dev_weight,
                               dev_c,
                               dev_scales,
                               dev_bias,
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
};

std::vector<int> problem_size = {m, n, k};
"""
    + tune_and_invoke_part
    + """
  return {c_out};
}

std::vector<std::vector<int64_t>> ${op_name}_InferShape(const std::vector<int64_t>& a_shape,
                                                        const std::vector<int64_t>& b_shape,
                                                        const std::vector<int64_t>& c_shape,
                                                        const std::vector<int64_t>& d_shape,
                                                        bool bool_trans_w) {
    if (bool_trans_w) {
        return {{a_shape[0], b_shape[0]}};
    } else {
        return {{a_shape[0], b_shape[1]}};
    }
}

std::vector<paddle::DataType> ${op_name}_InferDtype(const paddle::DataType& A_dtype) {
    return {A_dtype};
}

PD_BUILD_OP(${op_name})
    .Inputs({"x", "qweight", "scales", paddle::Optional("bias")})
    .Outputs({"out"})
    .SetKernelFn(PD_KERNEL(${op_name}_func))
    .Attrs({"bool_trans_w: bool"})
    .SetInferDtypeFn(PD_INFER_DTYPE(${op_name}_InferDtype))
    .SetInferShapeFn(PD_INFER_SHAPE(${op_name}_InferShape));
"""
)


def weight_only_int8(x, qweight, scales, bias=None, bool_trans_w=True):
    M, K = x.shape
    if bool_trans_w:
        N = qweight.shape[0]
        stride_bk = 1
        stride_bn = K
    else:
        N = qweight.shape[1]
        stride_bk = N
        stride_bn = 1

    if in_dynamic_or_pir_mode():
        assert x.is_contiguous(), ""
        assert qweight.is_contiguous(), ""

        # output = paddle.zeros((M, N), dtype=x.dtype)

        # grid = lambda META: (
        #     triton.cdiv(M, META['BLOCK_SIZE_M'])
        #     * triton.cdiv(N, META['BLOCK_SIZE_N']),
        #     META['SPLIT_K'],
        # )

        # wint8_kernel[grid](
        #     x,
        #     qweight,
        #     output,
        #     scales,
        #     bias,
        #     M,
        #     N,
        #     K,
        #     K,
        #     1,  # A always is rowmajor
        #     stride_bk,
        #     stride_bn,
        #     N,
        #     1,  # C always is rowmajor
        # )
        # return output

    op_name = "triton_wint8"
    if bool_trans_w:
        op_name = "triton_wint8_trans"

    # -1 means this value does not matter for triton compilation
    x_list = [-1, N, K, K, 1, stride_bk, stride_bn, N, 1]

    op_name = get_op_name_with_suffix(op_name, x_list)

    if (
        op_name in OpProtoHolder.instance().op_proto_map.keys()
        and in_dynamic_or_pir_mode()
    ):
        outs = _C_ops._run_custom_op(
            op_name, x, qweight, scales, bias, bool_trans_w
        )
        return outs[0]

    value_hint = get_value_hint(x_list)

    # output type is same as x, bias type is same as x
    dtypes = [x.dtype, qweight.dtype, x.dtype, scales.dtype, x.dtype]
    address_hint = get_pointer_hint(dtypes)
    assert (
        x.dtype == paddle.float16
    ), "weight_only_int8 now only support float16 as input x"

    python_package_name = f"{op_name}_package"

    generated_dir = (
        f"/zhoukangkang/2023-06-06minigpt/PaddleNLP/llm/inference/{op_name}"
    )
    os.makedirs(generated_dir, exist_ok=True)

    py_script_file = f"{generated_dir}/triton_kernels.py"
    extract_triton_kernel(wint8_kernel, py_script_file)

    op_dict = {"op_name": op_name, "reset_zero_when_tune": " "}
    # when tunning, we need to reset the out to zero.
    op_dict[
        "reset_zero_when_tune"
    ] = "cudaMemset((void*)dev_c, 0, sizeof(phi::dtype::float16) * m * n);"
    paddle_custom_op_file_path = f"{generated_dir}/{op_name}.cu"
    so_path = find_so_path(generated_dir, python_package_name)

    if so_path is None:
        with open(paddle_custom_op_file_path, "w") as f:
            f.write(SubstituteTemplate(triton_wint8_template, op_dict))
            f.close()

        # ahead of time compile command.
        aot_template = (
            f"""{python_path}   {compile_file} {py_script_file}   -n wint8_kernel   -o {generated_dir}/{op_name}_kernel --out-name {op_name}_kernel """
            + """ -w {num_warps}   -ns {num_stages}  \
        -s   "{address_hint} {value_hint}  {block_m},{block_n},{block_k}, {group_size_m}, {split_k}"   \
        -g   "((M+{block_m}-1)/{block_m}) * ((N+{block_n}-1)/{block_n}), {split_k}, 1" \
        """
        )

        codegen_commands = []
        for config in get_wint8_kernel_config():
            split_k = config.kwargs["SPLIT_K"]
            block_m = config.kwargs["BLOCK_SIZE_M"]
            block_n = config.kwargs["BLOCK_SIZE_N"]
            block_k = config.kwargs["BLOCK_SIZE_K"]
            group_size_m = config.kwargs["GROUP_SIZE_M"]
            num_stages = config.num_stages
            num_warps = config.num_warps

            if K % (split_k * block_k) != 0:
                print("config is not supported:", config)
                continue
            codegen_command = aot_template.format(
                address_hint=address_hint,
                value_hint=value_hint,
                num_warps=num_warps,
                num_stages=num_stages,
                block_m=block_m,
                block_n=block_n,
                block_k=block_k,
                split_k=split_k,
                group_size_m=group_size_m,
            )
            codegen_commands.append(codegen_command)

        multi_process_do(codegen_commands)

        link_command = f"{python_path}  {link_file}  {generated_dir}/*.h -o {generated_dir}/{op_name}_kernel"
        re = os.system(link_command)
        assert re == 0

        # rename the .c file to .cu
        rename_c_to_cu(generated_dir)
        # build the package to so, not install
        build_package(generated_dir, python_package_name)

    if op_name not in OpProtoHolder.instance().op_proto_map.keys():
        so_path = find_so_path(generated_dir, python_package_name)
        print("we find so_path: ", so_path)
        paddle.utils.cpp_extension.load_op_meta_info_and_register_op(so_path)

    if in_dynamic_or_pir_mode():
        outs = _C_ops._run_custom_op(
            op_name, x, qweight, scales, bias, bool_trans_w
        )
        return outs[0]

    helper = LayerHelper(op_name, **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    inputs = {
        'x': x,
        'qweight': qweight,
        'scales': scales,
        'bias@OPTIONAL': bias,
    }

    helper.append_op(
        type=op_name,
        inputs=inputs,
        attrs={"bool_trans_w": bool_trans_w},
        outputs={'out': out},
    )
    return out


########################### adaptive layer norm ###############################
@triton.jit
def adaptive_layer_norm_kernel(
    x_ptr,
    y_ptr,
    weight_ptr,
    bias_ptr,
    scale_ptr,
    shift_ptr,
    M,
    N,
    seq_size,
    epsilon,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(axis=0)
    x_ptr += row * N
    y_ptr += row * N
    # Compute mean
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for col_off in range(0, N, BLOCK_SIZE):
        cols = col_off + tl.arange(0, BLOCK_SIZE)
        eles = tl.load(x_ptr + cols, mask=cols < N, other=0.0).to(tl.float32)
        _mean += eles
    mean = tl.sum(_mean, axis=0) / N
    # Compute variance
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for col_off in range(0, N, BLOCK_SIZE):
        cols = col_off + tl.arange(0, BLOCK_SIZE)
        eles = tl.load(x_ptr + cols, mask=cols < N, other=0.0).to(tl.float32)
        eles = tl.where(cols < N, eles - mean, 0.0)
        _var += eles * eles
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + epsilon)
    # Compute output
    scale_ptr += (row // seq_size) * N
    shift_ptr += (row // seq_size) * N
    for col_off in range(0, N, BLOCK_SIZE):
        cols = col_off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        eles = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        x_hat = (eles - mean) * rstd
        if weight_ptr is not None:
            weights = tl.load(weight_ptr + cols, mask=mask, other=0.0)
            x_hat = x_hat * weights
        if bias_ptr is not None:
            bias = tl.load(bias_ptr + cols, mask=mask, other=0.0)
            x_hat = x_hat + bias
        scales = tl.load(scale_ptr + cols, mask=mask, other=0.0)
        shifts = tl.load(shift_ptr + cols, mask=mask, other=0.0)
        y = x_hat * (1 + scales) + shifts
        tl.store(y_ptr + cols, y, mask=mask)


triton_adaptive_layer_norm_template = (
    paddle_custom_op_head_part
    + """

std::vector<paddle::Tensor> ${op_name}_func(
    const paddle::Tensor &x,
    const paddle::Tensor &scale,
    const paddle::Tensor &shift,
    paddle::optional<paddle::Tensor> &weight,
    paddle::optional<paddle::Tensor> &bias,
    float epsilon) {
  int M = x.dims()[0] * x.dims()[1];
  int N = x.dims()[2];
  int seq_size = x.dims()[1];
  auto y = paddle::full(x.shape(), 0, x.dtype(), x.place());

  auto dev_x = get_tensor_ptr(x);
  auto dev_y = get_tensor_ptr(y);
  auto dev_scale = get_tensor_ptr(scale);
  auto dev_shift = get_tensor_ptr(shift);
  CUdeviceptr dev_weight = (CUdeviceptr)(nullptr);
  if (weight) {
    dev_weight = get_tensor_ptr(*weight);
  }
  CUdeviceptr dev_bias = (CUdeviceptr)(nullptr);
  if (bias) {
    dev_bias = get_tensor_ptr(*bias);
  }

  auto run_triton_kernel = [&](int algo_id) -> CUresult{
      return ${op_name}_kernel(y.stream(),
                                               dev_x,
                                               dev_y,
                                               dev_weight,
                                               dev_bias,
                                               dev_scale,
                                               dev_shift,
                                               M,
                                               N,
                                               seq_size,
                                               epsilon,
                                               algo_id);
  };

  std::vector<int> problem_size = {M, N};
"""
    + tune_and_invoke_part
    + """
  return {y};
}

std::vector<std::vector<int64_t>> ${op_name}_InferShape(
        const std::vector<int64_t>& A_shape) {
  return {A_shape};
}

std::vector<paddle::DataType> ${op_name}_InferDtype(const paddle::DataType& A_dtype) {
    return {A_dtype};
}

PD_BUILD_OP(${op_name})
    .Inputs({"x", "scale", "shift", paddle::Optional("weight"), paddle::Optional("bias")})
    .Outputs({"out"})
    .SetKernelFn(PD_KERNEL(${op_name}_func))
    .Attrs({"epsilon: float"})
    .SetInferDtypeFn(PD_INFER_DTYPE(${op_name}_InferDtype))
    .SetInferShapeFn(PD_INFER_SHAPE(${op_name}_InferShape));
"""
)


def adaptive_layer_norm(x, scale, shift, weight=None, bias=None, epsilon=1e-05):
    assert (
        len(x.shape) == 3
    ), "x should be 3-dim [batch_size, seq_size, feature_dim]"
    if weight is not None:
        assert len(weight.shape) == 1, "weight should be 1-dim [feature_dim]"
        assert (
            weight.shape[-1] == x.shape[-1]
        ), "x and weight should have same shape[-1] == feature_dim"
    if bias is not None:
        assert len(bias.shape) == 1, "bias should be 1-dim [feature_dim]"
        assert (
            bias.shape[-1] == x.shape[-1]
        ), "x and bias should have same shape[-1] == feature_dim"
    assert (
        len(scale.shape) == 2 and len(shift.shape) == 2
    ), "scale and shift should be 2-dim [batch_size, feature_dim]"
    assert (
        scale.shape[0] == shift.shape[0] == x.shape[0]
    ), "x, scale and shift should have same shape[0] == batch_size"
    assert (
        scale.shape[1] == shift.shape[1] == x.shape[-1]
    ), "x, scale and shift should have same shape[-1] == feature_dim"

    M = x.shape[0] * x.shape[1]
    N = x.shape[2]
    seq_size = x.shape[1]
    BLOCK_SIZE = min(1024, triton.next_power_of_2(N))

    # if in_dynamic_or_pir_mode():
    #     y = paddle.empty_like(x)
    #     adaptive_layer_norm_kernel[(M,)](
    #         x,
    #         y,
    #         weight,
    #         bias,
    #         scale,
    #         shift,
    #         M,
    #         N,
    #         seq_size,
    #         epsilon,
    #         BLOCK_SIZE=BLOCK_SIZE,
    #     )
    #     return y

    op_name = "triton_adaptive_layer_norm"
    if x.dtype == paddle.float16:
        op_name += "_fp16"
    elif x.dtype == paddle.float32:
        op_name += "_fp32"
    else:
        raise NotImplementedError(
            "triton_adaptive_layer_norm now supports only fp16 and fp32 dtype."
        )

    # if in_dynamic_or_pir_mode && op is already registered, call it directly.
    if (
        op_name in OpProtoHolder.instance().op_proto_map.keys()
        and in_dynamic_or_pir_mode()
    ):
        outs = _C_ops._run_custom_op(
            op_name, x, scale, shift, weight, bias, epsilon
        )
        return outs[0]

    x_list = [M, N, seq_size, epsilon]
    value_hint = get_value_hint(x_list)
    dtypes = [x.dtype] * 6
    address_hint = get_pointer_hint(dtypes)

    python_package_name = f"{op_name}_package"
    generated_dir = f"/tyk/Paddle/kai/triton/generated/{op_name}"
    os.makedirs(generated_dir, exist_ok=True)

    py_script_file = f"{generated_dir}/triton_kernels.py"
    extract_triton_kernel(adaptive_layer_norm_kernel, py_script_file)

    op_dict = {"op_name": op_name, "reset_zero_when_tune": ""}
    paddle_custom_op_file_path = f"{generated_dir}/{op_name}.cu"
    so_path = find_so_path(generated_dir, python_package_name)

    if so_path is None:
        print("== we do not find so_path, we need to compile it")
        with open(paddle_custom_op_file_path, "w") as f:
            f.write(
                SubstituteTemplate(triton_adaptive_layer_norm_template, op_dict)
            )
            f.close()

        # ahead of time compile command.
        aot_template = (
            f"""{python_path}   {compile_file} {py_script_file}   -n adaptive_layer_norm_kernel -o {generated_dir}/{op_name}_kernel --out-name {op_name}_kernel  """
            + """ -s "{address_hint} {value_hint}  {BLOCK_SIZE}"   \
                             -g "M, 1, 1" \
                       """
        )

        codegen_command = aot_template.format(
            address_hint=address_hint,
            value_hint=value_hint,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        re = os.system(codegen_command)
        assert re == 0

        link_command = f"{python_path}  {link_file}  {generated_dir}/*.h -o {generated_dir}/{op_name}_kernel"
        re = os.system(link_command)
        assert re == 0

        # rename the .c file to .cu
        rename_c_to_cu(generated_dir)
        # build the package to so, not install
        build_package(generated_dir, python_package_name)

    if op_name not in OpProtoHolder.instance().op_proto_map.keys():
        so_path = find_so_path(generated_dir, python_package_name)
        print("== we find so_path: ", so_path)
        assert so_path is not None
        paddle.utils.cpp_extension.load_op_meta_info_and_register_op(so_path)

    if in_dynamic_or_pir_mode():
        print(f"== we are in dynamic mode, op_name: {op_name}")
        outs = _C_ops._run_custom_op(
            op_name, x, scale, shift, weight, bias, epsilon
        )
        return outs[0]
    else:
        print(f"== we are in static mode, op_name: {op_name}")
        helper = LayerHelper(op_name, **locals())
        inputs = {
            'x': x,
            'scale': scale,
            'shift': shift,
            'weight@OPTIONAL': weight,
            'bias@OPTIONAL': bias,
        }
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
        helper.append_op(
            type=op_name,
            inputs=inputs,
            attrs={
                'epsilon': epsilon,
            },
            outputs={'out': out},
        )
        return out
