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

import os
import sys


def GenerateFileStructureForFinalDygraph(eager_dir):
    """
    paddle/fluid/eager
    |- generated
    |  |- CMakeLists.txt
    |  |  "add_subdirectory(forwards), add_subdirectory(backwards)"
    |
    |  |- forwards
    |     |- "dygraph_functions.cc"
    |     |- "dygraph_functions.h"
    |
    |  |- backwards
    |     |- "nodes.cc"
    |     |- "nodes.h"
    """
    # Directory Generation
    generated_dir = os.path.join(eager_dir, "api/generated/eager_generated")
    forwards_dir = os.path.join(generated_dir, "forwards")
    nodes_dir = os.path.join(generated_dir, "backwards")
    dirs = [generated_dir, forwards_dir, nodes_dir]
    for directory in dirs:
        if not os.path.exists(directory):
            os.mkdir(directory)

    # Empty files
    dygraph_forward_api_h_path = os.path.join(
        generated_dir, "dygraph_functions.h"
    )
    empty_files = [dygraph_forward_api_h_path]
    empty_files.append(os.path.join(forwards_dir, "dygraph_functions.cc"))
    empty_files.append(os.path.join(nodes_dir, "nodes.cc"))
    empty_files.append(os.path.join(nodes_dir, "nodes.h"))

    for path in empty_files:
        if not os.path.exists(path):
            open(path, 'a').close()


def GenerateFileStructureForIntermediateDygraph(eager_dir, split_count):
    """
    paddle/fluid/eager
    |- generated
    |  |- CMakeLists.txt
    |  |  "add_subdirectory(forwards), add_subdirectory(nodes)"
    |
    |  |- forwards
    |     |- "dygraph_forward_functions.cc"
    |     |- CMakeLists.txt
    |     |  "cc_library(dygraph_function SRCS dygraph_forward_functions.cc DEPS ${eager_deps} ${fluid_deps} GLOB_OP_LIB)"
    |
    |  |- nodes
    |     |- "nodes.cc"
    |     |- "nodes.h"
    |     |- CMakeLists.txt
    |     |  "cc_library(dygraph_node SRCS nodes.cc DEPS ${eager_deps} ${fluid_deps})"
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
    dygraph_forward_api_h_path = os.path.join(
        generated_dir, "dygraph_forward_api.h"
    )
    empty_files = [dygraph_forward_api_h_path]
    empty_files.append(os.path.join(nodes_dir, "nodes.h"))

    for i in range(split_count):
        empty_files.append(
            os.path.join(
                forwards_dir, "dygraph_forward_functions" + str(i + 1) + ".cc"
            )
        )
        empty_files.append(
            os.path.join(nodes_dir, "nodes" + str(i + 1) + ".cc")
        )
    empty_files.append(
        os.path.join(forwards_dir, "dygraph_forward_functions_args_info.cc")
    )
    empty_files.append(
        os.path.join(
            forwards_dir, "dygraph_forward_functions_args_type_info.cc"
        )
    )
    empty_files.append(
        os.path.join(forwards_dir, "dygraph_forward_functions_returns_info.cc")
    )
    for path in empty_files:
        if not os.path.exists(path):
            open(path, 'a').close()

    # CMakeLists
    nodes_level_cmakelist_path = os.path.join(nodes_dir, "CMakeLists.txt")
    generated_level_cmakelist_path = os.path.join(
        generated_dir, "CMakeLists.txt"
    )
    forwards_level_cmakelist_path = os.path.join(forwards_dir, "CMakeLists.txt")

    with open(nodes_level_cmakelist_path, "w") as f:
        f.write("add_custom_target(\n")
        f.write("  copy_dygraph_node\n")
        f.write(
            "  COMMAND ${CMAKE_COMMAND} -E copy_if_different \"${PADDLE_SOURCE_DIR}/paddle/fluid/eager/api/generated/fluid_generated/nodes/nodes.tmp.h\" \"${PADDLE_SOURCE_DIR}/paddle/fluid/eager/api/generated/fluid_generated/nodes/nodes.h\"\n"
        )
        for i in range(split_count):
            f.write(
                "  COMMAND ${CMAKE_COMMAND} -E copy_if_different \"${PADDLE_SOURCE_DIR}/paddle/fluid/eager/api/generated/fluid_generated/nodes/nodes"
                + str(i + 1)
                + ".tmp.cc\" \"${PADDLE_SOURCE_DIR}/paddle/fluid/eager/api/generated/fluid_generated/nodes/nodes"
                + str(i + 1)
                + ".cc\"\n"
            )

        f.write("  DEPENDS legacy_eager_codegen\n")
        f.write("  VERBATIM)\n")

        f.write("cc_library(dygraph_node SRCS ")
        for i in range(split_count):
            f.write("nodes" + str(i + 1) + ".cc ")
        f.write("${fluid_manual_nodes} DEPS ${eager_deps} ${fluid_deps})\n")
        f.write("add_dependencies(dygraph_node copy_dygraph_node)\n")

    with open(forwards_level_cmakelist_path, "w") as f:
        f.write("add_custom_target(\n")
        f.write("  copy_dygraph_forward_functions\n")
        f.write(
            "  COMMAND ${CMAKE_COMMAND} -E copy_if_different \"${PADDLE_SOURCE_DIR}/paddle/fluid/eager/api/generated/fluid_generated/dygraph_forward_api.tmp.h\" \"${PADDLE_SOURCE_DIR}/paddle/fluid/eager/api/generated/fluid_generated/dygraph_forward_api.h\"\n"
        )
        for i in range(split_count):
            f.write(
                "  COMMAND ${CMAKE_COMMAND} -E copy_if_different \"${PADDLE_SOURCE_DIR}/paddle/fluid/eager/api/generated/fluid_generated/forwards/dygraph_forward_functions"
                + str(i + 1)
                + ".tmp.cc\" \"${PADDLE_SOURCE_DIR}/paddle/fluid/eager/api/generated/fluid_generated/forwards/dygraph_forward_functions"
                + str(i + 1)
                + ".cc\"\n"
            )
        f.write(
            "  COMMAND ${CMAKE_COMMAND} -E copy_if_different \"${PADDLE_SOURCE_DIR}/paddle/fluid/eager/api/generated/fluid_generated/forwards/dygraph_forward_functions_args_info.tmp.cc\" \"${PADDLE_SOURCE_DIR}/paddle/fluid/eager/api/generated/fluid_generated/forwards/dygraph_forward_functions_args_info.cc\"\n"
        )
        f.write(
            "  COMMAND ${CMAKE_COMMAND} -E copy_if_different \"${PADDLE_SOURCE_DIR}/paddle/fluid/eager/api/generated/fluid_generated/forwards/dygraph_forward_functions_args_type_info.tmp.cc\" \"${PADDLE_SOURCE_DIR}/paddle/fluid/eager/api/generated/fluid_generated/forwards/dygraph_forward_functions_args_type_info.cc\"\n"
        )
        f.write(
            "  COMMAND ${CMAKE_COMMAND} -E copy_if_different \"${PADDLE_SOURCE_DIR}/paddle/fluid/eager/api/generated/fluid_generated/forwards/dygraph_forward_functions_returns_info.tmp.cc\" \"${PADDLE_SOURCE_DIR}/paddle/fluid/eager/api/generated/fluid_generated/forwards/dygraph_forward_functions_returns_info.cc\"\n"
        )
        f.write("  DEPENDS legacy_eager_codegen\n")
        f.write("  VERBATIM)\n")

        f.write("cc_library(dygraph_function SRCS ")
        for i in range(split_count):
            f.write("dygraph_forward_functions" + str(i + 1) + ".cc ")
        f.write("dygraph_forward_functions_args_info.cc ")
        f.write("dygraph_forward_functions_args_type_info.cc ")
        f.write("dygraph_forward_functions_returns_info.cc ")
        f.write(
            "${fluid_manual_functions} DEPS ${eager_deps} ${fluid_deps} ${GLOB_OP_LIB} ${GLOB_OPERATOR_DEPS})\n"
        )
        f.write(
            "add_dependencies(dygraph_function copy_dygraph_forward_functions)\n"
        )

    with open(generated_level_cmakelist_path, "w") as f:
        f.write("add_subdirectory(forwards)\nadd_subdirectory(nodes)")


if __name__ == "__main__":
    assert len(sys.argv) == 3
    eager_dir = sys.argv[1]
    split_count = int(sys.argv[2])
    GenerateFileStructureForIntermediateDygraph(eager_dir, split_count)
    GenerateFileStructureForFinalDygraph(eager_dir)
