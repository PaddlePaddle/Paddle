# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
"""
<<<<<<< HEAD
This script simply removes all grad ops and kernels. You should use this script
when cmake ON_INFER=ON, which can greatly reduce the volume of the prediction library.
"""

import glob
import os
import re
=======
This script simply removes all grad ops and kernels. You should use this script 
when cmake ON_INFER=ON, which can greatly reduce the volume of the prediction library.
"""

import os
import sys
import re
import glob
import io
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81


def find_type_files(cur_dir, file_type, file_list=[]):
    next_level_dirs = os.listdir(cur_dir)
    for next_level_name in next_level_dirs:
        next_level_dir = os.path.join(cur_dir, next_level_name)
        if os.path.isfile(next_level_dir):
            if os.path.splitext(next_level_dir)[1] == file_type:
                file_list.append(next_level_dir)
        elif os.path.isdir(next_level_dir):
            find_type_files(next_level_dir, file_type, file_list)
    return file_list


def find_kernel(content, pattern):
    res = re.findall(pattern, content, flags=re.DOTALL)
    ret = []
    for p in res:
        left, right = 0, 0
        for c in p:
            if c == '{':
                left += 1
            elif c == '}':
                right += 1

        if left == right:
            ret.append(p)

    return ret, len(ret)


def prune_phi_kernels():
    tool_dir = os.path.dirname(os.path.abspath(__file__))
<<<<<<< HEAD

    all_op = glob.glob(
        os.path.join(tool_dir, '../paddle/phi/kernels/**/*.cc'), recursive=True
    )
    all_op += glob.glob(
        os.path.join(tool_dir, '../paddle/phi/kernels/**/*.cu'), recursive=True
    )
=======
    if sys.version_info[0] == 3:
        all_op = glob.glob(os.path.join(tool_dir,
                                        '../paddle/phi/kernels/**/*.cc'),
                           recursive=True)
        all_op += glob.glob(os.path.join(tool_dir,
                                         '../paddle/phi/kernels/**/*.cu'),
                            recursive=True)
    elif sys.version_info[0] == 2:
        all_op = find_type_files(
            os.path.join(tool_dir, '../paddle/phi/kernels/'), '.cc')
        all_op = find_type_files(
            os.path.join(tool_dir, '../paddle/phi/kernels/'), '.cu', all_op)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    register_op_count = 0
    for op_file in all_op:
        need_continue = False
        file_blacklist = [
<<<<<<< HEAD
            "kernels/empty_kernel.cc",
            "/cast_kernel.c",
            "/batch_norm_kernel.c",
=======
            "kernels/empty_kernel.cc", "/cast_kernel.c", "/batch_norm_kernel.c"
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        ]
        for bname in file_blacklist:
            if op_file.find(bname) >= 0:
                need_continue = True
                break

        if need_continue:
            print("continue:", op_file)
            continue

        op_name = os.path.split(op_file)[1]
        all_matches = []
<<<<<<< HEAD
        with open(op_file, 'r', encoding='utf-8') as f:
            content = ''.join(f.readlines())
            op_pattern = r'PD_REGISTER_KERNEL\(.*?\).*?\{.*?\}'
=======
        with io.open(op_file, 'r', encoding='utf-8') as f:
            content = ''.join(f.readlines())
            op_pattern = 'PD_REGISTER_KERNEL\(.*?\).*?\{.*?\}'
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            op, op_count = find_kernel(content, op_pattern)
            register_op_count += op_count
            all_matches.extend(op)

        for p in all_matches:
            content = content.replace(p, '')

<<<<<<< HEAD
        with open(op_file, 'w', encoding='utf-8') as f:
            f.write(content)
=======
        with io.open(op_file, 'w', encoding='utf-8') as f:
            f.write(u'{}'.format(content))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    print('We erase all grad op and kernel for Paddle-Inference lib.')
    print('%50s%10s' % ('type', 'count'))
    print('%50s%10s' % ('REGISTER_OPERATOR', register_op_count))
    return True


def apply_patches():
    work_path = os.path.dirname(os.path.abspath(__file__)) + "/../"
    ret = os.system(
        "cd %s && rm -f paddle/fluid/inference/api/tensorrt_predictor.* "
        " && rm -f paddle/fluid/inference/api/paddle_tensorrt_predictor.h "
<<<<<<< HEAD
        " && git apply tools/infer_prune_patches/*.patch && cd -" % work_path
    )
=======
        " && git apply tools/infer_prune_patches/*.patch && cd -" % work_path)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    return ret == 0


def append_fluid_kernels():
    op_white_list = ["load", "load_combine"]

<<<<<<< HEAD
    # 1. add to makefile
    file_name = (
        os.path.dirname(os.path.abspath(__file__))
        + "/../paddle/fluid/inference/tensorrt/CMakeLists.txt"
    )
    append_str = "\nfile(APPEND ${pybind_file} \"USE_NO_KERNEL_OP__(tensorrt_engine);\\n\")\n"
    for op in op_white_list:
        append_str = (
            append_str
            + "file(APPEND ${pybind_file} \"USE_OP__(%s);\\n\")\n" % op
        )

    with open(file_name, 'r', encoding='utf-8') as f:
=======
    #1. add to makefile
    file_name = os.path.dirname(os.path.abspath(__file__)) \
                  + "/../paddle/fluid/inference/tensorrt/CMakeLists.txt"
    append_str = "\nfile(APPEND ${pybind_file} \"USE_NO_KERNEL_OP__(tensorrt_engine);\\n\")\n"
    for op in op_white_list:
        append_str = append_str + "file(APPEND ${pybind_file} \"USE_OP__(%s);\\n\")\n" % op

    with io.open(file_name, 'r', encoding='utf-8') as f:
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        content = ''.join(f.readlines())

    location_str = "nv_library(\n  tensorrt_op_teller\n  SRCS op_teller.cc\n  DEPS framework_proto device_context)"
    new_content = content.replace(location_str, location_str + append_str)

    if new_content == content:
<<<<<<< HEAD
        print(
            "ERROR: can not find \"%s\" in file \"%s\""
            % (location_str, file_name)
        )
        return False

    with open(file_name, 'w', encoding='utf-8') as f:
        f.write(new_content)

    # 2. add op and kernel register
    op_white_list.append("tensorrt_engine")
    tool_dir = os.path.dirname(os.path.abspath(__file__))
    all_op = glob.glob(
        os.path.join(tool_dir, '../paddle/fluid/operators/**/*.cc'),
        recursive=True,
    )
    all_op += glob.glob(
        os.path.join(tool_dir, '../paddle/fluid/operators/**/*.cu'),
        recursive=True,
    )

    for op_file in all_op:
        with open(op_file, 'r', encoding='utf-8') as f:
=======
        print("ERROR: can not find \"%s\" in file \"%s\"" %
              (location_str, file_name))
        return False

    with io.open(file_name, 'w', encoding='utf-8') as f:
        f.write(u'{}'.format(new_content))

    #2. add op and kernel register
    op_white_list.append("tensorrt_engine")
    tool_dir = os.path.dirname(os.path.abspath(__file__))
    if sys.version_info[0] == 3:
        all_op = glob.glob(os.path.join(tool_dir,
                                        '../paddle/fluid/operators/**/*.cc'),
                           recursive=True)
        all_op += glob.glob(os.path.join(tool_dir,
                                         '../paddle/fluid/operators/**/*.cu'),
                            recursive=True)
    elif sys.version_info[0] == 2:
        all_op = find_type_files(
            os.path.join(tool_dir, '../paddle/fluid/operators/'), '.cc')
        all_op = find_type_files(
            os.path.join(tool_dir, '../paddle/fluid/operators/'), '.cu', all_op)

    for op_file in all_op:
        with io.open(op_file, 'r', encoding='utf-8') as f:
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            content = ''.join(f.readlines())

        for op in op_white_list:
            patterns = {
<<<<<<< HEAD
                "REGISTER_OPERATOR": r"REGISTER_OPERATOR\(\s*%s\s*," % op,
                "REGISTER_OP_CPU_KERNEL": r"REGISTER_OP_CPU_KERNEL\(\s*%s\s*,"
                % op,
                "REGISTER_OP_CUDA_KERNEL": r"REGISTER_OP_CUDA_KERNEL\(\s*%s\s*,"
                % op,
=======
                "REGISTER_OPERATOR": "REGISTER_OPERATOR\(\s*%s\s*," % op,
                "REGISTER_OP_CPU_KERNEL":
                "REGISTER_OP_CPU_KERNEL\(\s*%s\s*," % op,
                "REGISTER_OP_CUDA_KERNEL":
                "REGISTER_OP_CUDA_KERNEL\(\s*%s\s*," % op
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            }
            for k, p in patterns.items():
                matches = re.findall(p, content, flags=re.DOTALL)
                if len(matches) > 0:
<<<<<<< HEAD
                    content = content.replace(
                        matches[0], matches[0].replace(k, k + "__")
                    )
                    with open(op_file, 'w', encoding='utf-8') as f:
                        f.write(content)
=======
                    content = content.replace(matches[0],
                                              matches[0].replace(k, k + "__"))
                    with io.open(op_file, 'w', encoding='utf-8') as f:
                        f.write(u'{}'.format(content))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    return True


if __name__ == '__main__':

    print("================ step 1: apply patches =======================")
<<<<<<< HEAD
    assert apply_patches()
    print("==============================================================\n")

    print("================ step 2: append fluid op/kernels==============")
    assert append_fluid_kernels()
    print("==============================================================\n")

    print("================ step 3:prune phi kernels ====================")
    assert prune_phi_kernels()
=======
    assert (apply_patches())
    print("==============================================================\n")

    print("================ step 2: append fluid op/kernels==============")
    assert (append_fluid_kernels())
    print("==============================================================\n")

    print("================ step 3:prune phi kernels ====================")
    assert (prune_phi_kernels())
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    print("==============================================================\n")
