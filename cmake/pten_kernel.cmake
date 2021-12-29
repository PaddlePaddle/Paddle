# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# call kernel_declare need to make sure whether the target of input exists
function(kernel_declare TARGET_LIST)
    foreach(kernel_path ${TARGET_LIST})
        file(READ ${kernel_path} kernel_impl)
        # TODO(chenweihang): rename PT_REGISTER_CTX_KERNEL to PT_REGISTER_KERNEL
        # NOTE(chenweihang): now we don't recommend to use digit in kernel name
        string(REGEX MATCH "(PT_REGISTER_CTX_KERNEL|PT_REGISTER_GENERAL_KERNEL)\\([ \t\r\n]*[a-z0-9_]*," first_registry "${kernel_impl}")
        if (NOT first_registry STREQUAL "")
            # parse the first kernel name
            string(REPLACE "PT_REGISTER_CTX_KERNEL(" "" kernel_name "${first_registry}")
            string(REPLACE "PT_REGISTER_GENERAL_KERNEL(" "" kernel_name "${kernel_name}")
            string(REPLACE "," "" kernel_name "${kernel_name}")
            string(REGEX REPLACE "[ \t\r\n]+" "" kernel_name "${kernel_name}")
            # append kernel declare into declarations.h
            # TODO(chenweihang): default declare ALL_LAYOUT for each kernel
            if (${kernel_path} MATCHES "./cpu\/")
                file(APPEND ${kernel_declare_file} "PT_DECLARE_KERNEL(${kernel_name}, CPU, ALL_LAYOUT);\n")
            elseif (${kernel_path} MATCHES "./gpu\/")
                file(APPEND ${kernel_declare_file} "PT_DECLARE_KERNEL(${kernel_name}, GPU, ALL_LAYOUT);\n")
            elseif (${kernel_path} MATCHES "./xpu\/")
                file(APPEND ${kernel_declare_file} "PT_DECLARE_KERNEL(${kernel_name}, XPU, ALL_LAYOUT);\n")
            else ()
                # deal with device independent kernel, now we use CPU temporaary
                file(APPEND ${kernel_declare_file} "PT_DECLARE_KERNEL(${kernel_name}, CPU, ALL_LAYOUT);\n")
            endif()
        endif()
    endforeach()
endfunction()

function(kernel_library TARGET)
    set(common_srcs)
    set(cpu_srcs)
    set(gpu_srcs)
    set(xpu_srcs)
    # parse and save the deps kerenl targets
    set(all_srcs)
    set(kernel_deps)

    set(oneValueArgs "")
    set(multiValueArgs SRCS DEPS)
    cmake_parse_arguments(kernel_library "${options}" "${oneValueArgs}"
        "${multiValueArgs}" ${ARGN})

    list(LENGTH kernel_library_SRCS kernel_library_SRCS_len)
    # one kernel only match one impl file in each backend
    if (${kernel_library_SRCS_len} EQUAL 0)
        if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${TARGET}.cc)
            list(APPEND common_srcs ${CMAKE_CURRENT_SOURCE_DIR}/${TARGET}.cc)
        endif()
        if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/cpu/${TARGET}.cc)
            list(APPEND cpu_srcs ${CMAKE_CURRENT_SOURCE_DIR}/cpu/${TARGET}.cc)
        endif()
        if (WITH_GPU OR WITH_ROCM)
            if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/gpu/${TARGET}.cu)
                list(APPEND gpu_srcs ${CMAKE_CURRENT_SOURCE_DIR}/gpu/${TARGET}.cu)
            endif()
        endif()
        if (WITH_XPU)
            if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/xpu/${TARGET}.cc)
                list(APPEND xpu_srcs ${CMAKE_CURRENT_SOURCE_DIR}/xpu/${TARGET}.cc)
            endif()
        endif()
    else()
        # TODO(chenweihang): impl compile by source later
    endif()

    list(APPEND all_srcs ${CMAKE_CURRENT_SOURCE_DIR}/${TARGET}.h)
    list(APPEND all_srcs ${common_srcs})
    list(APPEND all_srcs ${cpu_srcs})
    list(APPEND all_srcs ${gpu_srcs})
    list(APPEND all_srcs ${xpu_srcs})
    foreach(src ${all_srcs})
        file(READ ${src} target_content)
        string(REGEX MATCHALL "#include \"paddle\/pten\/kernels\/[a-z0-9_]+_kernel.h\"" include_kernels ${target_content})
        foreach(include_kernel ${include_kernels})
            string(REGEX REPLACE "#include \"paddle\/pten\/kernels\/" "" kernel_name ${include_kernel})
            string(REGEX REPLACE ".h\"" "" kernel_name ${kernel_name})
            list(APPEND kernel_deps ${kernel_name})
        endforeach()
    endforeach()
    list(REMOVE_DUPLICATES kernel_deps)
    list(REMOVE_ITEM kernel_deps ${TARGET})

    list(LENGTH common_srcs common_srcs_len)
    list(LENGTH cpu_srcs cpu_srcs_len)
    list(LENGTH gpu_srcs gpu_srcs_len)
    list(LENGTH xpu_srcs xpu_srcs_len)

    if (${common_srcs_len} GREATER 0)
        # If the kernel has a device independent public implementation,
        # we will use this implementation and will not adopt the implementation
        # under specific devices
        if (WITH_GPU)
            nv_library(${TARGET} SRCS ${common_srcs} DEPS ${kernel_library_DEPS} ${kernel_deps})
        elseif (WITH_ROCM)
            hip_library(${TARGET} SRCS ${common_srcs} DEPS ${kernel_library_DEPS} ${kernel_deps})
        else()
            cc_library(${TARGET} SRCS ${common_srcs} DEPS ${kernel_library_DEPS} ${kernel_deps})
        endif()
    else()
        # If the kernel has a header file declaration, but no corresponding
        # implementation can be found, this is not allowed
        if (${cpu_srcs_len} EQUAL 0 AND ${gpu_srcs_len} EQUAL 0 AND
            ${xpu_srcs_len} EQUAL 0)
            message(FATAL_ERROR "Cannot find any implementation for ${TARGET}")
        else()
            if (WITH_GPU)
                if (${cpu_srcs_len} GREATER 0 OR ${gpu_srcs_len} GREATER 0)
                    nv_library(${TARGET} SRCS ${cpu_srcs} ${gpu_srcs} DEPS ${kernel_library_DEPS} ${kernel_deps})
                endif()
            elseif (WITH_ROCM)
                if (${cpu_srcs_len} GREATER 0 OR ${gpu_srcs_len} GREATER 0)
                    hip_library(${TARGET} SRCS ${cpu_srcs} ${gpu_srcs} DEPS ${kernel_library_DEPS} ${kernel_deps})
                endif()
            else()
                if (${cpu_srcs_len} GREATER 0 OR ${xpu_srcs_len} GREATER 0)
                    cc_library(${TARGET} SRCS ${cpu_srcs} ${xpu_srcs} DEPS ${kernel_library_DEPS} ${kernel_deps})
                endif()
            endif()
        endif()
    endif()

    if (${common_srcs_len} GREATER 0 OR ${cpu_srcs_len} GREATER 0 OR
        ${gpu_srcs_len} GREATER 0 OR ${xpu_srcs_len} GREATER 0)
        # append target into PTEN_KERNELS property
        get_property(pten_kernels GLOBAL PROPERTY PTEN_KERNELS)
        set(pten_kernels ${pten_kernels} ${TARGET})
        set_property(GLOBAL PROPERTY PTEN_KERNELS ${pten_kernels})
    endif()

    # parse kernel name and auto generate kernel declaration
    # here, we don't need to check WITH_XXX, because if not WITH_XXX, the
    # xxx_srcs_len will be equal to 0
    if (${common_srcs_len} GREATER 0)
        kernel_declare(${common_srcs})
    endif()
    if (${cpu_srcs_len} GREATER 0)
        kernel_declare(${cpu_srcs})
    endif()
    if (${gpu_srcs_len} GREATER 0)
        kernel_declare(${gpu_srcs})
    endif()
    if (${xpu_srcs_len} GREATER 0)
        kernel_declare(${xpu_srcs})
    endif()
endfunction()

function(register_kernels)
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs EXCLUDES DEPS)
    cmake_parse_arguments(register_kernels "${options}" "${oneValueArgs}"
        "${multiValueArgs}" ${ARGN})

    file(GLOB KERNELS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*_kernel.h")
    string(REPLACE ".h" "" KERNELS "${KERNELS}")
    list(LENGTH register_kernels_DEPS register_kernels_DEPS_len)

    foreach(target ${KERNELS})
        list(FIND register_kernels_EXCLUDES ${target} _index)
        if (${_index} EQUAL -1)
            if (${register_kernels_DEPS_len} GREATER 0)
                kernel_library(${target} DEPS ${register_kernels_DEPS})
            else()
                kernel_library(${target})
            endif()
        endif()
    endforeach()
endfunction()
