#!/bin/python

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

import argparse
import json
import yaml
from typing import List, Dict, Any


def parse_args():
    parser = argparse.ArgumentParser("gather phi kernel and infermate info")
    parser.add_argument(
        "--paddle_root_path",
        type=str,
        required=True,
        help="root path of paddle src[WORK_PATH/Paddle].")
    parser.add_argument(
        "--kernel_info_file",
        type=str,
        required=True,
        help="kernel info file generated by get_phi_kernel_function.sh.")
    parser.add_argument(
        "--infermeta_wrap_file",
        type=str,
        required=True,
        help="inferMeta wrap info file.")
    parser.add_argument(
        "--attr_info_file", type=str, required=True, help="attr info file.")
    parser.add_argument(
        "--generate_file",
        type=str,
        required=True,
        default="../paddle/infrt/kernel/phi/infershaped/infershaped_kernel_launchers.cc",
        help="generated file.")
    args = parser.parse_args()
    return args


def get_api_yaml_info(file_path):
    f = open(file_path + "/python/paddle/utils/code_gen/api.yaml", "r")
    cont = f.read()
    return yaml.load(cont, Loader=yaml.FullLoader)


def get_kernel_info(file_path):
    f = open(file_path, "r")
    cont = f.readlines()
    return [l.strip() for l in cont]


def get_attr_info(file_path):
    """
    phi_gpu.argsort.float64.any $axisBool$descending
    """
    ret = {}
    with open(file_path, 'r') as f:
        cont = f.readlines()
        for l in cont:
            datas = l.strip().split(' ')
            if len(datas) == 2:
                attrs = datas[1].split('$')
                ret[datas[0]] = attrs[1:]
            else:
                ret[datas[0]] = None
    return ret


def merge(infer_meta_data, kernel_data, wrap_data):
    meta_map = {}
    for api in infer_meta_data:
        if "kernel" not in api or "infer_meta" not in api:
            continue
        meta_map[api["kernel"]["func"]] = api["infer_meta"]["func"]
    wrap_map = {}
    for l in wrap_data:
        wrap_map[l.split()[0]] = l.split()[1]

    full_kernel_data = []
    for l in kernel_data:
        key = l.split()[0]
        if key in meta_map:
            if key in meta_map:
                full_kernel_data.append((l + " " + wrap_map[key]).split())
            else:
                full_kernel_data.append((l + " " + meta_map[key]).split())
        else:
            full_kernel_data.append((l + " unknown").split())

    return full_kernel_data


def gen_warn_info():
    return """// Generated by tools/infrt/gen_phi_kernel_register.py for infrt.
// DO NOT edit or include it within paddle.
"""


def gen_include_headers():
    return """
#include "paddle/infrt/kernel/phi/infershaped/infershaped_kernel_launchers.h"
#include "paddle/infrt/kernel/phi/infershaped/phi_kernel_launcher.h"
#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/include/kernels.h"
#include "paddle/phi/include/infermeta.h"
#include "paddle/phi/infermeta/generated.h"
"""


def gen_namespace():
    return ("""
namespace infrt {
namespace kernel {

""", """

}  // namespace kernel
}  // namespace infrt
""")


def gen_context(val):
    if val == "CPU":
        return "phi::CPUContext", "phi_cpu"
    elif val == "GPU":
        return "phi::GPUContext", "phi_gpu"
    # elif val == "XPU":
    #     return "phi::XPUContext", "phi_xpu"
    else:
        # raise Exception(f"Unknown context type {val}")
        return "", ""


def gen_layout(val):
    if val == "ALL_LAYOUT":
        return 'any'
    else:
        # TODO(wilber): now only process ALL_LAYOUT
        raise Exception(f"Unknown layout type {val}")


def gen_kernel_func(val, ctx_name, dtype_name):
    if '<' in val and '>' in val:
        st = val.index('<')
        ed = val.index('>')
        func_name = val[:st]
        template_name = val[st + 1:ed]
        if 'phi::' in template_name:
            return "&phi::" + val
        else:
            return "&phi::" + func_name + "<phi::" + template_name + ">"
    else:
        return "&phi::" + val + "<" + dtype_name + ", " + ctx_name + ">"


def gen_dtype(vals: List[str]):
    ir_dtypes, origin_dtypes = [], []
    for val in vals:
        if val == "float":
            ir_dtypes.append("float32")
            origin_dtypes.append("float")
        elif val == "double":
            ir_dtypes.append("float64")
            origin_dtypes.append("double")
        elif val == "float16":
            ir_dtypes.append("float16")
            origin_dtypes.append("paddle::experimental::float16")
        elif val == "bfloat16":
            ir_dtypes.append("bf16")
            origin_dtypes.append("paddle::experimental::bfloat16")
        elif val == "bool":
            ir_dtypes.append("bool")
            origin_dtypes.append("bool")
        elif val == "int8_t":
            ir_dtypes.append("int8")
            origin_dtypes.append("int8_t")
        elif val == "uint8_t":
            ir_dtypes.append("uint8")
            origin_dtypes.append("uint8_t")
        elif val == "int16_t":
            ir_dtypes.append("int16")
            origin_dtypes.append("int16_t")
        elif val == "int" or val == "int32_t":
            ir_dtypes.append("int32")
            origin_dtypes.append("int32_t")
        elif val == "int64_t":
            ir_dtypes.append("int64")
            origin_dtypes.append("int64_t")
        elif val == "complex<float>" or val == "complex64":
            ir_dtypes.append("complex64")
            origin_dtypes.append("paddle::experimental::complex64")
        elif val == "complex<double>" or val == "complex128":
            ir_dtypes.append("complex128")
            origin_dtypes.append("paddle::experimental::complex128")
        elif val == "pstring":
            ir_dtypes.append("pstring")
            origin_dtypes.append("paddle::experimental::pstring")
        elif val == "ALL_DTYPE":
            ir_dtypes.append("all")
            origin_dtypes.append("all")
        else:
            if "VA_ARGS" in val:
                continue
            raise Exception(f"Unknown data type {val}")
    return ir_dtypes, origin_dtypes


# Note: Now only process CPUContext and GPUContext.


def gen_register_code_info(item: List[str], attr_data: Dict[str, List[str]]):
    """
    item: ['add', 'CPU', 'ALL_LAYOUT', 'AddKernel', 'float', 'double', '...'(varaidic types), 'ElementwiseInferMeta']
    attr_data: {'phi_cpu.arg_min.float32.any': ['axisBool', 'keepdimsBool', 'flatten', 'dtype']}
    """
    ctx_name, ir_ctx_name = gen_context(item[1])
    if (ctx_name == ""):
        return ""
    item[2] = gen_layout(item[2])
    ir_dtypes, origin_dtypes = gen_dtype(item[4:-1])
    infer_shape_func = "&phi::" + item[-1]

    res = ""

    if item[-1] == "unknown":
        # TODO(wilber): handle the unknown inferShape func.
        return ""

    for ir_dtype, origin_dtype in zip(ir_dtypes, origin_dtypes):
        kernel_func = gen_kernel_func(item[3], ctx_name, origin_dtype)
        ir_name = ir_ctx_name + '.' + item[0].lower(
        ) + '.' + ir_dtype + '.' + item[2].lower()
        if ir_name in attr_data.keys() and attr_data[ir_name] is not None:
            attr_names = ', '.join(
                ["\"" + a + "\"" for a in attr_data[ir_name]])
            res += f"""
registry->AddKernelWithAttrs("{ir_name}","""

            res += f"""
    std::bind(&KernelLauncherFunc<decltype({kernel_func}),
                                  {kernel_func},
                                  decltype({infer_shape_func}),
                                  {infer_shape_func}>,
              KernelLauncher<decltype({kernel_func}),
                                  {kernel_func},
                                  decltype({infer_shape_func}),
                                  {infer_shape_func}>(),
              std::placeholders::_1),
    {{{attr_names}}});
"""

        else:
            res += f"""
registry->AddKernel("{ir_name}","""

            res += f"""
    std::bind(&KernelLauncherFunc<decltype({kernel_func}),
                                  {kernel_func},
                                  decltype({infer_shape_func}),
                                  {infer_shape_func}>,
              KernelLauncher<decltype({kernel_func}),
                                  {kernel_func},
                                  decltype({infer_shape_func}),
                                  {infer_shape_func}>(),
              std::placeholders::_1));
"""

    return res


def gen_register_info(resources: List[List[str]],
                      attr_data: Dict[str, List[str]]):
    """
    resources: [['add', 'CPU', 'ALL_LAYOUT', 'AddKernel', 'float', 'double', '...'(varaidic types), 'ElementwiseInferMeta'], ...]
    attr_data: {'phi_cpu.arg_min.float32.any': ['axisBool', 'keepdimsBool', 'flatten', 'dtype']}
    """
    res = "void RegisterInferShapeLaunchers(host_context::KernelRegistry* registry) {"

    # register cpu kernels.
    for item in resources:
        # The output string is polluted by C++ macros, here the \ is removed
        update_item = [v.strip('\\') for v in item]
        if update_item[1] != "CPU":
            continue
        code = gen_register_code_info(item, attr_data)
        if (code == ""):
            continue
        res += code

    # register gpu kernels.
    res += "\n#ifdef INFRT_WITH_GPU"
    for item in resources:
        # The output string is polluted by C++ macros, here the \ is removed
        update_item = [v.strip('\\') for v in item]
        if update_item[1] != "GPU":
            continue
        code = gen_register_code_info(item, attr_data)
        if (code == ""):
            continue
        res += code
    res += "#endif // INFRT_WITH_GPU"

    res += "\n}"
    return res


def gen_phi_kernel_register_code(resources: List[List[str]],
                                 attr_data: Dict[str, List[str]],
                                 src_file_path: str):
    source_file = open(src_file_path, 'w')
    source_file.write(gen_warn_info())
    source_file.write(gen_include_headers())
    namespace = gen_namespace()
    source_file.write(namespace[0])
    source_file.write(gen_register_info(resources, attr_data))
    source_file.write(namespace[1])
    source_file.close()


if __name__ == "__main__":
    args = parse_args()
    infer_meta_data = get_api_yaml_info(args.paddle_root_path)
    kernel_data = get_kernel_info(args.kernel_info_file)
    info_meta_wrap_data = get_kernel_info(args.infermeta_wrap_file)
    attr_data = get_attr_info(args.attr_info_file)
    out = merge(infer_meta_data, kernel_data, info_meta_wrap_data)
    gen_phi_kernel_register_code(out, attr_data, args.generate_file)
