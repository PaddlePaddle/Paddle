# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import sys
import os

if __name__ == "__main__":
    assert len(sys.argv) == 2
    eager_dir = sys.argv[1]

    op_list = []
    with open(f"{eager_dir}/auto_code_generator/op_list.txt", "r") as f:
        for line in f:
            line = str(line.strip())
            op_list.append(line)
    """
    paddle/fluid/eager
    |- generated
    |  |- CMakeLists.txt
    |  |  "add_subdirectory(forwards), add_subdirectory(nodes)"
    |  
    |  |- forwards
    |     |- op_name + "_dygraph.cc"
    |     |- CMakeLists.txt
    |     |  "cc_library(dygraph_function SRCS op_name+"_dygraph.cc" DEPS ${eager_deps} ${fluid_deps} GLOB_OP_LIB)"
    |
    |  |- nodes
    |     |- op_name + "_node.cc"
    |     |- op_name + "_node.h"
    |     |- CMakeLists.txt
    |     |  "cc_library(dygraph_node SRCS op_name+"_node.cc" DEPS ${eager_deps} ${fluid_deps})"
    | 
    |  |- dygraph_forward_api.h
    """
    # Directory Generation
    generated_dir = os.path.join(eager_dir, "api/generated/fluid_generated")
    forwards_dir = os.path.join(generated_dir, "forwards")
    nodes_dir = os.path.join(generated_dir, "nodes")
    dirs = [generated_dir, forwards_dir, nodes_dir]
    for directory in dirs:
        if not os.path.exists(directory):
            os.mkdir(directory)

    # Empty files
    dygraph_forward_api_h_path = os.path.join(generated_dir,
                                              "dygraph_forward_api.h")
    empty_files = [dygraph_forward_api_h_path]
    for op_name in op_list:
        empty_files.append(os.path.join(forwards_dir, op_name + "_dygraph.cc"))
        empty_files.append(os.path.join(nodes_dir, op_name + "_node.cc"))
        empty_files.append(os.path.join(nodes_dir, op_name + "_node.h"))

    for path in empty_files:
        if not os.path.exists(path):
            open(path, 'a').close()

    # CMakeLists
    nodes_level_cmakelist_path = os.path.join(nodes_dir, "CMakeLists.txt")
    generated_level_cmakelist_path = os.path.join(generated_dir,
                                                  "CMakeLists.txt")
    forwards_level_cmakelist_path = os.path.join(forwards_dir, "CMakeLists.txt")

    with open(nodes_level_cmakelist_path, "w") as f:
        f.write(
            "cc_library(dygraph_node SRCS %s DEPS ${eager_deps} ${fluid_deps})\n"
            % " ".join([op_name + '_node.cc' for op_name in op_list]))
        f.write("add_dependencies(dygraph_node eager_codegen)")

    with open(forwards_level_cmakelist_path, "w") as f:
        f.write(
            "cc_library(dygraph_function SRCS %s DEPS ${eager_deps} ${fluid_deps} ${GLOB_OP_LIB})\n"
            % " ".join([op_name + '_dygraph.cc' for op_name in op_list]))
        f.write("add_dependencies(dygraph_function eager_codegen)")

    with open(generated_level_cmakelist_path, "w") as f:
        f.write("add_subdirectory(forwards)\nadd_subdirectory(nodes)")
