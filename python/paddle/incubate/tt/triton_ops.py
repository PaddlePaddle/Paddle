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
    get_dtype_str,
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


class KernelInterface:
    def __init__(
        self,
        func,
        custom_op_template,
        other_config,
        key_args=["1"],
    ):
        self.func = func
        self.key_args = key_args

        import inspect

        signature = inspect.signature(func)
        self.arg_names = [v.name for v in signature.parameters.values()]
        for ele in self.arg_names:
            assert self.arg_names.count(ele) == 1
        arg_defaults = [v.default for v in signature.parameters.values()]

        # self.annotations = {
        #     name: ty for name, ty in func.__annotations__.items()
        # }
        self.annotations = dict(func.__annotations__)

        self.constexprs = [
            self.arg_names.index(name)
            for name in self.arg_names
            if self.annotations.get(name) == triton.language.core.constexpr
        ]

        self.arg_exclude_constexpr = [
            self.arg_names[i]
            for i in range(len(self.arg_names))
            if i not in self.constexprs
        ]

        import textwrap

        py_script = textwrap.dedent(inspect.getsource(func))

        import re

        pat = r"def\s" + func.__name__
        func_begin = re.findall(pat, py_script)
        assert len(func_begin) == 1
        func_begin = func_begin[0]
        py_script = py_script[py_script.find(func_begin) :]

        def decorator(*args, **kwargs):
            all_input = []

            for i in range(len(args)):
                all_input.append(args[i])

            position_arguments_num = len(all_input)
            for i in range(position_arguments_num, len(self.arg_names)):
                if self.arg_names[i] in kwargs.keys():
                    all_input.append(kwargs[self.arg_names[i]])
                else:
                    # means this input is not specified, it muse be a tl.constexpr.
                    assert i in self.constexprs
                    all_input.append(None)

            dtypes = []
            x_list = []
            const_args = [self.arg_names[i] for i in self.constexprs]
            # we dont allow there are two strings in const_args, and one is a substring of the other.
            for i in const_args:
                for j in const_args:
                    if i != j and i.find(j) != -1:
                        raise ValueError(
                            f"We find {i}, {j} in tl.constexpr args, and {j} is a substring of {i}, please modify your triton kernel arguments names to avoid this."
                        )

            const_hint_dict = {}
            for i in range(len(all_input)):
                ele = all_input[i]
                if (
                    type(ele) == paddle.Tensor
                    or type(ele) == paddle.base.framework.EagerParamBase
                    or type(ele) == paddle.base.framework.Parameter
                    or type(ele) == paddle.base.framework.Variable
                ):
                    dtypes.append(ele.dtype)
                elif i in self.constexprs:
                    const_hint_dict[self.arg_names[i]] = ele
                else:
                    x_list.append(ele)

            op_name = self.op_name

            python_package_name = f"{op_name}_package"

            generated_dir = os.getenv("TRITON_KERNEL_CACHE_DIR", None)
            print("the kernel cache dir is:", generated_dir)
            assert (
                generated_dir is not None
            ), "TRITON_KERNEL_CACHE_DIR is None, please set it such as export TRITON_KERNEL_CACHE_DIR=/tmp/haha "
            generated_dir = f"{generated_dir}/{op_name}"
            os.makedirs(generated_dir, exist_ok=True)

            py_script_file = f"{generated_dir}/triton_kernels.py"
            extract_triton_kernel(func, py_script_file)

            address_hint = get_pointer_hint(dtypes)
            value_hint = get_value_hint(x_list)
            const_args = [f"{{{ele}}}" for ele in const_args]
            const_args = ",".join(const_args)

            lanuch_grid = list(self.grid)
            for i in range(len(lanuch_grid)):
                ele = lanuch_grid[i]
                if type(ele) == str:
                    for key in const_hint_dict.keys():
                        if key in ele:
                            ele = ele.replace(key, f"{{{key}}}")
                else:
                    ele = str(ele)

                lanuch_grid[i] = ele
            if len(lanuch_grid) < 3:
                lanuch_grid += ['1'] * (3 - len(lanuch_grid))
            lanuch_grid = ",".join(lanuch_grid)

            op_dict = {"op_name": op_name, "reset_zero_when_tune": ""}
            op_dict["triton_kernel_args"] = ",".join(self.arg_exclude_constexpr)
            op_dict["key"] = ",".join(self.key_args)
            # when tunning, we need to reset the out to zero.
            if "reset_zero_when_tune" in other_config.keys():
                op_dict["reset_zero_when_tune"] = other_config[
                    "reset_zero_when_tune"
                ]

            paddle_custom_op_file_path = f"{generated_dir}/{op_name}.cu"
            so_path = find_so_path(generated_dir, python_package_name)

            if so_path is None:
                print("== we do not find so_path, we need to compile it")
                with open(paddle_custom_op_file_path, "w") as f:
                    f.write(
                        SubstituteTemplate(
                            paddle_custom_op_head_part + custom_op_template,
                            op_dict,
                        )
                    )
                    f.close()

                # ahead of time compile command.
                aot_template = (
                    f"""{python_path}   {compile_file} {py_script_file}   -n {func.__name__} -o {generated_dir}/{op_name}_kernel --out-name {op_name}_kernel  """
                    + """ -w {num_warps} -ns {num_stages} """
                    + f""" -s"{address_hint} {value_hint} {const_args}" """
                    + f"""  -g "{lanuch_grid}" """
                )
                all_tune_config = list(self.tune_config)
                if len(all_tune_config) == 0:
                    # when user do not specify config, we use const_hint_dict as config.
                    all_tune_config = [const_hint_dict]
                    # reset const_hint_dict as empty.
                    const_hint_dict = {}
                codegen_commands = []
                for config in all_tune_config:
                    for key in const_hint_dict.keys():
                        if const_hint_dict[key] is not None:
                            if key not in config.keys():
                                config[key] = const_hint_dict[key]
                            else:
                                raise ValueError(
                                    f"you specify {key} both in arguments and config, this is wrong."
                                )
                        else:
                            assert (
                                key in config.keys()
                            ), f"you must specify {key} in your config."
                    if "num_warps" not in config.keys():
                        config['num_warps'] = 4
                    if "num_stages" not in config.keys():
                        config['num_stages'] = 4

                    for key in config:
                        assert (
                            config[key] is not None
                        ), f"{key} must be specified."
                    codegen_command = aot_template.format(
                        **config,
                    )
                    print(codegen_command)
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
                print("== we find so_path: ", so_path)
                assert so_path is not None
                paddle.utils.cpp_extension.load_op_meta_info_and_register_op(
                    so_path
                )

        self.decorator = decorator

    def __getitem__(self, op_name_and_grid):
        assert len(op_name_and_grid) >= 2, "len(op_name_and_grid) must >= 2."
        self.op_name = op_name_and_grid[0]
        self.grid = op_name_and_grid[1]
        if len(op_name_and_grid) == 2:
            self.tune_config = {}
        else:
            self.tune_config = op_name_and_grid[2]
        return self.decorator


def paddle_use_triton(custom_op_template, other_config={}, key=[]):
    def decorator(func):
        return KernelInterface(func, custom_op_template, other_config, key)

    return decorator


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
                            {
                                "SPLIT_K": split_k,
                                "BLOCK_SIZE_M": block_m,
                                "BLOCK_SIZE_N": block_n,
                                "BLOCK_SIZE_K": block_k,
                                "GROUP_SIZE_M": 8,
                                "num_stages": num_stages,
                                "num_warps": num_warps,
                            }
                        )
    return configs


triton_wint8_template = (
    """
std::vector<paddle::Tensor> ${op_name}_func(
    const paddle::Tensor& x,
    const paddle::Tensor& qweight,
    const paddle::Tensor& scales,
    paddle::optional<paddle::Tensor>& bias,
    bool bool_trans_w) {
  int M = x.shape()[0];
  int K = x.shape()[1];
  int N = scales.shape()[0];

  auto c_out = paddle::full({M, N}, 0, x.dtype(), x.place());

  auto a_ptr = get_tensor_ptr(x);
  auto b_ptr = get_tensor_ptr(qweight);
  auto c_ptr = get_tensor_ptr(c_out);
  auto bs_ptr = get_tensor_ptr(scales);
  CUdeviceptr bias_ptr = (CUdeviceptr)(nullptr);
  if (bias) {
    bias_ptr = get_tensor_ptr(*bias);
  }

  int stride_bk = N;
  int stride_bn = 1;

  if (bool_trans_w) {
    stride_bk = 1;
    stride_bn = K;
  }
  int stride_am = K;
  int stride_ak = 1;

  int stride_cm = N;
  int stride_cn = 1;

  auto run_stream = c_out.stream();
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

wint8_kernel_other_config = {
    "reset_zero_when_tune": "cudaMemset((void*)c_ptr, 0, sizeof(phi::dtype::float16) * M * N);"
}


@paddle_use_triton(
    custom_op_template=triton_wint8_template,
    other_config=wint8_kernel_other_config,
    key=["M", "N", "K"],
)
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


def weight_only_int8(x, qweight, scales, bias=None, bool_trans_w=True):
    '''
    Examples:

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    import paddle
    from paddle.nn.quant import weight_quantize, weight_only_linear

    M = 16
    N = 4096
    K = 4096*4

    activation = paddle.randn((M, K), dtype=paddle.float16)
    original_weight = paddle.randn((K, N), dtype=paddle.float16)
    bias = paddle.rand((N,), dtype=paddle.float16) * 10
    triton_scale = paddle.max(paddle.abs(original_weight), axis=0) / 127

    perm_qweight, scale = weight_quantize(original_weight, algo="weight_only_int8")

    assert paddle.max(triton_scale - scale) == 0

    # 下面是paddle的cutlass代码
    import datetime
    for i in range(100):
        paddle_cutlass_output = weight_only_linear(activation, perm_qweight, bias, scale)

    paddle.device.synchronize()
    starttime = datetime.datetime.now()
    for i in range(100):
        paddle_cutlass_output = weight_only_linear(activation, perm_qweight, bias, scale)
    paddle.device.synchronize()
    endtime = datetime.datetime.now()
    duringtime = endtime - starttime
    time_ms = duringtime.seconds * 1000 + duringtime.microseconds / 1000.0
    print("paddle cutlass The whoel end to end time : ", time_ms, "ms")

    # 下面是triton的计算代码
    bool_trans_w_triton = False

    triton_qweight = original_weight / triton_scale.reshape([1, N])
    triton_qweight = paddle.round(triton_qweight)
    triton_qweight = paddle.clip(triton_qweight, min=-127, max=127)
    triton_qweight = triton_qweight.astype("int8")

    if bool_trans_w_triton:
        triton_qweight = triton_qweight.transpose([1,0]).contiguous()

    assert activation.is_contiguous()
    assert triton_qweight.is_contiguous()
    assert scale.is_contiguous()
    triton_uint_qweight = (triton_qweight.astype("int32") + 128).astype("uint8")

    for i in range(100):
        triton_output = paddle.incubate.tt.weight_only_int8(
            activation,
            triton_uint_qweight,
            triton_scale,
            bias, bool_trans_w=bool_trans_w_triton)

    paddle.device.synchronize()

    starttime = datetime.datetime.now()
    for i in range(100):
        triton_output = paddle.incubate.tt.weight_only_int8(
            activation,
            triton_uint_qweight,
            triton_scale,
            bias,
            bool_trans_w = bool_trans_w_triton)
    paddle.device.synchronize()
    endtime = datetime.datetime.now()
    duringtime = endtime - starttime
    time_ms = duringtime.seconds * 1000 + duringtime.microseconds / 1000.0
    print("triton The whoel end to end time : ", time_ms, "ms")

    if bool_trans_w_triton:
        triton_qweight = triton_qweight.transpose([1,0]).contiguous()

    for i in range(100):
        dequantized_weight = triton_qweight.astype("float16") * scale.reshape([1, N])
        baseline = paddle.matmul(activation, dequantized_weight)
        baseline += bias

    paddle.device.synchronize()
    starttime = datetime.datetime.now()

    for i in range(100):
        dequantized_weight = triton_qweight.astype("float16") * scale.reshape([1, N])
        baseline = paddle.matmul(activation, dequantized_weight)
        baseline += bias
    paddle.device.synchronize()
    endtime = datetime.datetime.now()
    duringtime = endtime - starttime
    time_ms = duringtime.seconds * 1000 + duringtime.microseconds / 1000.0
    print("baseline The whoel end to end time : ", time_ms, "ms")

    print("triton and baseline max diff", paddle.max(paddle.abs(triton_output - baseline)))
    print("triton and cutlass max diff", paddle.max(paddle.abs(triton_output - paddle_cutlass_output)))
    '''

    M, K = x.shape
    if bool_trans_w:
        N = qweight.shape[0]
        stride_bk = 1
        stride_bn = K
    else:
        N = qweight.shape[1]
        stride_bk = N
        stride_bn = 1

    op_name = "triton_wint8"
    if bool_trans_w:
        op_name = "triton_wint8_trans"

    # -1 means this value does not matter for triton compilation
    x_list = [-1, N, K, K, 1, stride_bk, stride_bn, N, 1]

    op_name = get_op_name_with_suffix(op_name, x_list)

    if op_name not in OpProtoHolder.instance().op_proto_map.keys():
        assert x.is_contiguous(), ""
        assert qweight.is_contiguous(), ""

        # below code is register this kernel, will not run this kernel.
        output = paddle.zeros((1, 1), dtype=x.dtype)

        grid = (
            "((M+BLOCK_SIZE_M-1)/BLOCK_SIZE_M) * ((N+BLOCK_SIZE_N-1)/BLOCK_SIZE_N)",
            'SPLIT_K',
        )

        wint8_kernel[(op_name, grid, get_wint8_kernel_config())](
            x,
            qweight,
            output,
            scales,
            bias,
            M,
            N,
            K,
            K,
            1,  # A always is rowmajor
            stride_bk,
            stride_bn,
            N,
            1,  # C always is rowmajor
        )

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
fused_adaLN_scale_residual_template = (
    """


std::vector<paddle::Tensor> ${op_name}_func(
    const paddle::Tensor &x,
    const paddle::Tensor &mha_out,
    const paddle::Tensor &gate_msa,
    const paddle::Tensor &scale_mlp,
    const paddle::Tensor &shift_mlp,
    paddle::optional<paddle::Tensor> &weight,
    paddle::optional<paddle::Tensor> &bias,
    float epsilon) {
  int M = x.dims()[0] * x.dims()[1];
  int N = x.dims()[2];
  int seq_size = x.dims()[1];
  auto resi_out = paddle::empty(x.shape(), x.dtype(), x.place());
  auto adaLN_out = paddle::empty(x.shape(), x.dtype(), x.place());

  auto x_ptr = get_tensor_ptr(x);
  auto mha_out_ptr = get_tensor_ptr(mha_out);
  auto resi_out_ptr = get_tensor_ptr(resi_out);
  auto adaLN_out_ptr = get_tensor_ptr(adaLN_out);
  auto gate_msa_ptr = get_tensor_ptr(gate_msa);
  auto scale_mlp_ptr = get_tensor_ptr(scale_mlp);
  auto shift_mlp_ptr = get_tensor_ptr(shift_mlp);
  CUdeviceptr weight_ptr = (CUdeviceptr)(nullptr);
  if (weight) {
    weight_ptr = get_tensor_ptr(*weight);
  }
  CUdeviceptr bias_ptr = (CUdeviceptr)(nullptr);
  if (bias) {
    bias_ptr = get_tensor_ptr(*bias);
  }
  auto  run_stream = adaLN_out.stream();
"""
    + tune_and_invoke_part
    + """
    return {resi_out, adaLN_out};
}

std::vector<std::vector<int64_t>> ${op_name}_InferShape(
        const std::vector<int64_t>& A_shape) {
  return {A_shape, A_shape};
}

std::vector<paddle::DataType> ${op_name}_InferDtype(const paddle::DataType& A_dtype) {
    return {A_dtype, A_dtype};
}

PD_BUILD_OP(${op_name})
    .Inputs({"x", "mha_out", "gate_msa", "scale_mlp", "shift_mlp", paddle::Optional("weight"), paddle::Optional("bias")})
    .Outputs({"resi_out", "adaLN_out"})
    .SetKernelFn(PD_KERNEL(${op_name}_func))
    .Attrs({"epsilon: float"})
    .SetInferDtypeFn(PD_INFER_DTYPE(${op_name}_InferDtype))
    .SetInferShapeFn(PD_INFER_SHAPE(${op_name}_InferShape));
"""
)


@paddle_use_triton(
    custom_op_template=fused_adaLN_scale_residual_template,
    key=["M"],
)
def fused_adaLN_scale_residual_kernel(
    x_ptr,  # input: residual input of attention
    mha_out_ptr,  # input: attention result
    gate_msa_ptr,
    scale_mlp_ptr,
    shift_mlp_ptr,
    weight_ptr,
    bias_ptr,
    resi_out_ptr,  # output: residual result of attention
    adaLN_out_ptr,  # output: adaptive layer norm result
    M,
    N,
    seq_size,
    epsilon,
    N_npo2: tl.constexpr,
):
    row = tl.program_id(axis=0)
    mha_out_ptr += row * N
    x_ptr += row * N
    resi_out_ptr += row * N
    adaLN_out_ptr += row * N
    gate_msa_ptr += (row // seq_size) * N
    scale_mlp_ptr += (row // seq_size) * N
    shift_mlp_ptr += (row // seq_size) * N

    all_offs = tl.arange(0, N_npo2)
    all_mask = all_offs < N
    # compute residual
    mha_eles = tl.load(mha_out_ptr + all_offs, mask=all_mask, other=0.0).to(
        tl.float32
    )
    x_eles = tl.load(x_ptr + all_offs, mask=all_mask, other=0.0).to(tl.float32)
    gate_msa_eles = tl.load(gate_msa_ptr + all_offs, mask=all_mask, other=0.0)

    _resi_outs = mha_eles * gate_msa_eles + x_eles
    tl.store(resi_out_ptr + all_offs, _resi_outs, mask=all_mask)

    # compute mean var
    mean = tl.sum(_resi_outs, axis=0) / N
    var = tl.sum(_resi_outs * _resi_outs, axis=0) / N - mean * mean
    rstd = 1 / tl.sqrt(var + epsilon)

    # compute adaLN
    resi_hat = (_resi_outs - mean) * rstd
    if weight_ptr is not None:
        weights = tl.load(weight_ptr + all_offs, mask=all_mask, other=0.0)
        resi_hat = resi_hat * weights
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + all_offs, mask=all_mask, other=0.0)
        resi_hat = resi_hat + bias
    scales = tl.load(scale_mlp_ptr + all_offs, mask=all_mask, other=0.0)
    shifts = tl.load(shift_mlp_ptr + all_offs, mask=all_mask, other=0.0)
    y = resi_hat * (1 + scales) + shifts
    tl.store(adaLN_out_ptr + all_offs, y, mask=all_mask)


def fused_adaLN_scale_residual(
    x,
    mha_out,
    gate_msa,
    scale_mlp,
    shift_mlp,
    weight=None,
    bias=None,
    epsilon=1e-05,
):
    '''
    Examples:

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    import paddle

    batch = 2
    seq = 3600
    hidd = 4096
    dtype= "float16"
    epsilon = 1e-5
    x = paddle.rand([batch, seq, hidd], dtype=dtype)
    mha_out = paddle.rand([batch, seq, hidd], dtype=dtype)
    weight = paddle.rand([hidd], dtype=dtype)
    bias = paddle.rand([hidd], dtype=dtype)

    gate_msa = paddle.rand([batch, hidd], dtype=dtype)
    scale_mlp_x = paddle.rand([batch, hidd], dtype=dtype)
    shift_mlp_x = paddle.rand([batch, hidd], dtype=dtype)


    def modulate(x, shift, scale):
        return x * (1 + scale.unsqueeze(axis=1)) + shift.unsqueeze(axis=1)


    def paddle_fused_adaLN(x, mha_out, gate, hidd, scale, shift, weight, bias, epsilon):
        resi_out_paddle = mha_out * gate.unsqueeze(axis=1) + x
        layer_norm_out_paddle = paddle.nn.functional.layer_norm(resi_out_paddle, [hidd], weight, bias, epsilon)
        adaLN_out_paddle = modulate(layer_norm_out_paddle, shift, scale).to(dtype)
        return resi_out_paddle, adaLN_out_paddle


    for i in range(100):
        resi_out_triton, adaLN_out_triton = paddle.incubate.tt.fused_adaLN_scale_residual(x, mha_out, gate_msa, scale_mlp_x, shift_mlp_x, weight, bias, epsilon)

    for i in range(100):
        resi_out_paddle, adaLN_out_paddle = paddle_fused_adaLN(x, mha_out, gate_msa, hidd, scale_mlp_x, shift_mlp_x, weight, bias, epsilon)

    print("adaLN_maxdiff: ", paddle.max(paddle.abs(adaLN_out_paddle - adaLN_out_triton)))
    print("resi_maxdiff: ", paddle.max(paddle.abs(resi_out_paddle - resi_out_triton)))
    '''

    assert x.shape == mha_out.shape, "x and mha_out should have same shape"
    assert (
        gate_msa.shape == scale_mlp.shape == shift_mlp.shape
    ), "gate_msa, scale_mlp and shift_mlp should have same shape"

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
        len(scale_mlp.shape) == 2 and len(shift_mlp.shape) == 2
    ), "scale and shift should be 2-dim [batch_size, feature_dim]"
    assert (
        scale_mlp.shape[0] == shift_mlp.shape[0] == x.shape[0]
    ), "x, scale and shift should have same shape[0] == batch_size"
    assert (
        scale_mlp.shape[1] == shift_mlp.shape[1] == x.shape[-1]
    ), "x, scale and shift should have same shape[-1] == feature_dim"

    M = x.shape[0] * x.shape[1]
    N = x.shape[2]
    seq_size = x.shape[1]
    N_npo2 = triton.next_power_of_2(N)

    op_name = "triton_fused_adaLN_scale_residual"
    op_name += get_dtype_str(x.dtype)
    op_name += f"_{N_npo2}"

    fused_adaLN_scale_residual_kernel_config = [
        {'num_warps': 1},
    ]

    if op_name not in OpProtoHolder.instance().op_proto_map.keys():
        resi_out = paddle.empty_like(x)
        adaLN_out = paddle.empty_like(x)
        grid = ("M",)
        fused_adaLN_scale_residual_kernel[
            (op_name, grid, fused_adaLN_scale_residual_kernel_config)
        ](
            x,
            mha_out,
            gate_msa,
            scale_mlp,
            shift_mlp,
            weight,
            weight,
            resi_out,
            adaLN_out,
            M,
            N,
            seq_size,
            epsilon,
            N_npo2=N_npo2,
        )

    if in_dynamic_or_pir_mode():
        print(f"== we are in dynamic mode, op_name: {op_name}")
        outs = _C_ops._run_custom_op(
            op_name,
            x,
            mha_out,
            gate_msa,
            scale_mlp,
            shift_mlp,
            weight,
            bias,
            epsilon,
        )
        return outs[0], outs[1]
    else:
        print(f"== we are in dynamic to static mode, op_name: {op_name}")
        helper = LayerHelper(op_name, **locals())
        inputs = {
            'x': x,
            'mha_out': mha_out,
            'gate_msa': gate_msa,
            'scale_mlp': scale_mlp,
            'shift_mlp': shift_mlp,
            'weight@OPTIONAL': weight,
            'bias@OPTIONAL': bias,
        }
        resi_out = helper.create_variable_for_type_inference(dtype=x.dtype)
        adaLN_out = helper.create_variable_for_type_inference(dtype=x.dtype)
        helper.append_op(
            type=op_name,
            inputs=inputs,
            attrs={
                'epsilon': epsilon,
            },
            outputs={'resi_out': resi_out, 'adaLN_out': adaLN_out},
        )
        return resi_out, adaLN_out


triton_adaptive_layer_norm_template = (
    """

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
  auto y = paddle::empty(x.shape(), x.dtype(), x.place());

  auto x_ptr = get_tensor_ptr(x);
  auto y_ptr = get_tensor_ptr(y);
  auto scale_ptr = get_tensor_ptr(scale);
  auto shift_ptr = get_tensor_ptr(shift);
  CUdeviceptr weight_ptr = (CUdeviceptr)(nullptr);
  if (weight) {
    weight_ptr = get_tensor_ptr(*weight);
  }
  CUdeviceptr bias_ptr = (CUdeviceptr)(nullptr);
  if (bias) {
    bias_ptr = get_tensor_ptr(*bias);
  }
  auto run_stream = y.stream();
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


@paddle_use_triton(
    custom_op_template=triton_adaptive_layer_norm_template,
    key=["M"],
)
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
    _sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    _sum_square = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for col_off in range(0, N, BLOCK_SIZE):
        cols = col_off + tl.arange(0, BLOCK_SIZE)
        eles = tl.load(x_ptr + cols, mask=cols < N, other=0.0).to(tl.float32)
        _sum += eles
        _sum_square += eles * eles
    mean = tl.sum(_sum, axis=0) / N
    var = tl.sum(_sum_square, axis=0) / N - mean * mean
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


def adaptive_layer_norm(x, scale, shift, weight=None, bias=None, epsilon=1e-05):
    '''
    Examples:

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    import paddle

    def modulate(x, shift, scale):
        return x * (1 + scale.unsqueeze(axis=1)) + shift.unsqueeze(axis=1)

    batch = 2
    seq = 3600
    hidd = 4096
    dtype= "float16"
    x = paddle.rand([batch, seq, hidd], dtype=dtype)
    weight = paddle.rand([hidd], dtype=dtype)
    bias = paddle.rand([hidd], dtype=dtype)

    shift_msa_x = paddle.rand([batch, hidd], dtype=dtype)
    scale_msa_x = paddle.rand([batch, hidd], dtype=dtype)

    for i in range(100):
        mt_result = paddle.incubate.tt.adaptive_layer_norm(x, scale_msa_x, shift_msa_x, weight, bias)

    for i in range(100):
        baseline = modulate(paddle.nn.functional.layer_norm(x, [hidd], weight, bias, 1e-5), shift_msa_x, scale_msa_x)

    print(paddle.max(paddle.abs(baseline-mt_result)))

    '''

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

    op_name = "triton_adaptive_layer_norm"
    op_name += get_dtype_str(x.dtype)
    op_name += f"_{BLOCK_SIZE}"

    adaptive_layer_norm_kernel_config = [
        {'num_warps': 1},
    ]

    if op_name not in OpProtoHolder.instance().op_proto_map.keys():
        y = paddle.empty_like(x)
        grid = ("M",)
        adaptive_layer_norm_kernel[
            (op_name, grid, adaptive_layer_norm_kernel_config)
        ](
            x,
            y,
            y,
            y,
            y,
            y,
            M,
            N,
            seq_size,
            epsilon,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    if in_dynamic_or_pir_mode():
        print(f"== we are in dynamic mode, op_name: {op_name}")
        outs = _C_ops._run_custom_op(
            op_name, x, scale, shift, weight, bias, epsilon
        )
        return outs[0]
    else:
        print(f"== we are in dynamic to static mode, op_name: {op_name}")
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


rms_norm_template = (
    """

std::vector<paddle::Tensor> ${op_name}_func(
    const paddle::Tensor &x,
    paddle::optional<paddle::Tensor> &weight,
    paddle::optional<paddle::Tensor> &bias,
    float epsilon) {
  int M = x.dims()[0] * x.dims()[1] * x.dims()[2];
  int N = x.dims()[3];
  auto y = paddle::empty(x.shape(), x.dtype(), x.place());

  auto x_ptr = get_tensor_ptr(x);
  auto y_ptr = get_tensor_ptr(y);
  CUdeviceptr weight_ptr = (CUdeviceptr)(nullptr);
  if (weight) {
    weight_ptr = get_tensor_ptr(*weight);
  }
  CUdeviceptr bias_ptr = (CUdeviceptr)(nullptr);
  if (bias) {
    bias_ptr = get_tensor_ptr(*bias);
  }
  auto run_stream = y.stream();
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
    .Inputs({"x", paddle::Optional("weight"), paddle::Optional("bias")})
    .Outputs({"out"})
    .SetKernelFn(PD_KERNEL(${op_name}_func))
    .Attrs({"epsilon: float"})
    .SetInferDtypeFn(PD_INFER_DTYPE(${op_name}_InferDtype))
    .SetInferShapeFn(PD_INFER_SHAPE(${op_name}_InferShape));
"""
)


@paddle_use_triton(
    custom_op_template=rms_norm_template,
    key=["M"],
)
def rms_norm_kernel(
    x_ptr,
    y_ptr,
    weight_ptr,
    bias_ptr,
    M,
    N,
    epsilon,
    BLOCK_SIZE_M: tl.constexpr,
    N_npo2: tl.constexpr,
):
    row = tl.program_id(axis=0)

    offs_am = tl.arange(0, BLOCK_SIZE_M)
    offs_an = tl.arange(0, N_npo2)

    # compute var
    all_offs = (row * BLOCK_SIZE_M + offs_am[:, None]) % M * N + offs_an[
        None, :
    ]

    x_eles = tl.load(x_ptr + all_offs, mask=offs_an[None, :] < N, other=0.0).to(
        tl.float32
    )
    var = tl.sum(x_eles * x_eles, axis=1) / N

    resi_hat = x_eles / tl.sqrt(var[:, None] + epsilon)

    if weight_ptr is not None:
        weights = tl.load(weight_ptr + offs_an, mask=offs_an < N, other=0.0)
        resi_hat = resi_hat * weights

    if bias_ptr is not None:
        bias = tl.load(bias_ptr + offs_an, mask=offs_an < N, other=0.0)
        resi_hat = resi_hat + bias

    tl.store(y_ptr + all_offs, resi_hat, mask=offs_an[None, :] < N)


def rms_norm(x, weight=None, bias=None, epsilon=1e-05):
    '''
    Examples:

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    import paddle

    batch = 2
    seq = 3600
    num_heads = 1
    head_dim = 64*30
    dtype= "float16"
    x = paddle.rand([batch, seq, num_heads, head_dim], dtype=dtype)
    weight = paddle.rand([head_dim], dtype=dtype)
    bias = paddle.rand([head_dim], dtype=dtype)

    for i in range(100):
        baseline = paddle.incubate.nn.functional.fused_rms_norm(x, weight, bias, 1e-5, begin_norm_axis=3)

    for i in range(100):
        mt_result = paddle.incubate.tt.rms_norm(x,weight,bias,1e-5)


    baseline = baseline[0]
    print(paddle.max(paddle.abs(baseline-mt_result)))

    '''

    assert len(x.shape) == 4, "x should be 4-dim."
    if weight is not None:
        assert len(weight.shape) == 1, "weight should be 1-dim"
        assert (
            weight.shape[-1] == x.shape[-1]
        ), "x and weight should have same shape[-1]"
    if bias is not None:
        assert len(bias.shape) == 1, "bias should be 1-dim"
        assert (
            bias.shape[-1] == x.shape[-1]
        ), "x and bias should have same shape[-1]"

    M = x.shape[0] * x.shape[1] * x.shape[2]
    N = x.shape[3]
    N_npo2 = triton.next_power_of_2(N)

    op_name = "triton_rms_norm"
    op_name += get_dtype_str(x.dtype)
    op_name += f"_{N_npo2}"

    rms_norm_kernel_config = []
    if N_npo2 <= 64:
        rms_norm_kernel_config.append({'BLOCK_SIZE_M': 4, 'num_warps': 1})
    else:
        rms_norm_kernel_config.append({'BLOCK_SIZE_M': 1, 'num_warps': 4})

    if op_name not in OpProtoHolder.instance().op_proto_map.keys():
        y = paddle.empty_like(x)
        grid = ("((M+BLOCK_SIZE_M-1)/BLOCK_SIZE_M)",)
        rms_norm_kernel[(op_name, grid, rms_norm_kernel_config)](
            x,
            y,
            weight,
            x,
            -1,  # M,
            N,
            epsilon,
            BLOCK_SIZE_M=4,
            N_npo2=N_npo2,
        )

    if in_dynamic_or_pir_mode():
        print(f"== we are in dynamic mode, op_name: {op_name}")
        outs = _C_ops._run_custom_op(op_name, x, weight, bias, epsilon)
        return outs[0]
    else:
        print(f"== we are in dynamic to static mode, op_name: {op_name}")
        helper = LayerHelper(op_name, **locals())
        inputs = {
            'x': x,
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
