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

function(generate_unify_header DIR_NAME)
    set(options "")
    set(oneValueArgs HEADER_NAME SKIP_SUFFIX)
    set(multiValueArgs "")
    cmake_parse_arguments(generate_unify_header "${options}" "${oneValueArgs}"
        "${multiValueArgs}" ${ARGN})

    # get header name and suffix
    set(header_name "${DIR_NAME}")
    list(LENGTH generate_unify_header_HEADER_NAME generate_unify_header_HEADER_NAME_len)
    if(${generate_unify_header_HEADER_NAME_len} GREATER 0)
        set(header_name "${generate_unify_header_HEADER_NAME}")
    endif()
    set(skip_suffix "")
    list(LENGTH generate_unify_header_SKIP_SUFFIX generate_unify_header_SKIP_SUFFIX_len)
    if(${generate_unify_header_SKIP_SUFFIX_len} GREATER 0)
        set(skip_suffix "${generate_unify_header_SKIP_SUFFIX}")
    endif()

    # generate target header file
    set(header_file ${CMAKE_CURRENT_SOURCE_DIR}/include/${header_name}.h)
    file(WRITE ${header_file} "// Header file generated by paddle/pten/CMakeLists.txt for external users,\n// DO NOT edit or include it within paddle.\n\n#pragma once\n\n")

    # get all top-level headers and write into header file
    file(GLOB HEADERS "${CMAKE_CURRENT_SOURCE_DIR}\/${DIR_NAME}\/*.h")
    foreach(header ${HEADERS})
        if("${skip_suffix}" STREQUAL "")
            string(REPLACE "${PADDLE_SOURCE_DIR}\/" "" header "${header}")
            file(APPEND ${header_file} "#include \"${header}\"\n")
        else()
            string(FIND "${header}" "${skip_suffix}.h" skip_suffix_found)
            if(${skip_suffix_found} EQUAL -1)
                string(REPLACE "${PADDLE_SOURCE_DIR}\/" "" header "${header}")
                file(APPEND ${header_file} "#include \"${header}\"\n")
            endif()
        endif()
    endforeach()
    # append header into extension.h
    string(REPLACE "${PADDLE_SOURCE_DIR}\/" "" header_file "${header_file}")
    file(APPEND ${pten_extension_header_file} "#include \"${header_file}\"\n")
endfunction()

# call kernel_declare need to make sure whether the target of input exists
function(kernel_declare TARGET_LIST)
    foreach(kernel_path ${TARGET_LIST})
        file(READ ${kernel_path} kernel_impl)
        # TODO(chenweihang): rename PT_REGISTER_KERNEL to PT_REGISTER_KERNEL
        # NOTE(chenweihang): now we don't recommend to use digit in kernel name
        string(REGEX MATCH "(PT_REGISTER_KERNEL|PT_REGISTER_GENERAL_KERNEL)\\([ \t\r\n]*[a-z0-9_]*," first_registry "${kernel_impl}")
        if (NOT first_registry STREQUAL "")
            # parse the first kernel name
            string(REPLACE "PT_REGISTER_KERNEL(" "" kernel_name "${first_registry}")
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
            elseif (${kernel_path} MATCHES "./mkldnn\/")
                file(APPEND ${kernel_declare_file} "PT_DECLARE_KERNEL(${kernel_name}, MKLDNN, ALL_LAYOUT);\n")
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
    set(mkldnn_srcs)
    set(selected_rows_srcs)
    # parse and save the deps kerenl targets
    set(all_srcs)
    set(kernel_deps)

    set(oneValueArgs SUB_DIR)
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
        if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/selected_rows/${TARGET}.cc)
            list(APPEND selected_rows_srcs ${CMAKE_CURRENT_SOURCE_DIR}/selected_rows/${TARGET}.cc)
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
        if (WITH_MKLDNN)
            if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/mkldnn/${TARGET}.cc)
                list(APPEND mkldnn_srcs ${CMAKE_CURRENT_SOURCE_DIR}/mkldnn/${TARGET}.cc)
            endif()
        endif()
    else()
        # TODO(chenweihang): impl compile by source later
    endif()

    list(APPEND all_srcs ${CMAKE_CURRENT_SOURCE_DIR}/${TARGET}.h)
    if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/impl/${TARGET}_impl.h)
        list(APPEND all_srcs ${CMAKE_CURRENT_SOURCE_DIR}/impl/${TARGET}_impl.h)
    endif()
    list(APPEND all_srcs ${common_srcs})
    list(APPEND all_srcs ${cpu_srcs})
    list(APPEND all_srcs ${gpu_srcs})
    list(APPEND all_srcs ${xpu_srcs})
    list(APPEND all_srcs ${mkldnn_srcs})
    foreach(src ${all_srcs})
        file(READ ${src} target_content)
        string(REGEX MATCHALL "#include \"paddle\/pten\/kernels\/[a-z0-9_]+_kernel.h\"" include_kernels ${target_content})
        if ("${kernel_library_SUB_DIR}" STREQUAL "")
            string(REGEX MATCHALL "#include \"paddle\/pten\/kernels\/[a-z0-9_]+_kernel.h\"" include_kernels ${target_content})
        else()
            string(REGEX MATCHALL "#include \"paddle\/pten\/kernels\/${kernel_library_SUB_DIR}\/[a-z0-9_]+_kernel.h\"" include_kernels ${target_content})
        endif()
        foreach(include_kernel ${include_kernels})
        if ("${kernel_library_SUB_DIR}" STREQUAL "")
            string(REGEX REPLACE "#include \"paddle\/pten\/kernels\/" "" kernel_name ${include_kernel})
        else()
            string(REGEX REPLACE "#include \"paddle\/pten\/kernels\/${kernel_library_SUB_DIR}\/" "" kernel_name ${include_kernel})
        endif()
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
    list(LENGTH mkldnn_srcs mkldnn_srcs_len)
    list(LENGTH selected_rows_srcs selected_rows_srcs_len)

    # Build Target according different src organization
    if((${cpu_srcs_len} GREATER 0 OR ${gpu_srcs_len} GREATER 0 OR
        ${xpu_srcs_len} GREATER 0 OR ${mkldnn_srcs_len} GREATER 0) AND (${common_srcs_len} GREATER 0 OR 
        ${selected_rows_srcs_len} GREATER 0))
        # If the common_srcs/selected_rows_srcs depends on specific device srcs, build target using this rule.
        if (WITH_GPU)
            if (${cpu_srcs_len} GREATER 0 OR ${gpu_srcs_len} GREATER 0 OR ${mkldnn_srcs_len} GREATER 0)
                nv_library(${TARGET}_part SRCS ${cpu_srcs} ${gpu_srcs} ${mkldnn_srcs} DEPS ${kernel_library_DEPS} ${kernel_deps})
                nv_library(${TARGET} SRCS ${common_srcs} ${selected_rows_srcs} DEPS ${TARGET}_part)
            endif()
        elseif (WITH_ROCM)
            if (${cpu_srcs_len} GREATER 0 OR ${gpu_srcs_len} GREATER 0 OR ${mkldnn_srcs_len} GREATER 0)
                hip_library(${TARGET}_part SRCS ${cpu_srcs} ${gpu_srcs} ${mkldnn_srcs} DEPS ${kernel_library_DEPS} ${kernel_deps})
                hip_library(${TARGET} SRCS ${common_srcs} ${selected_rows_srcs} DEPS ${TARGET}_part)
            endif()
        else()
            if (${cpu_srcs_len} GREATER 0 OR ${xpu_srcs_len} GREATER 0 OR ${mkldnn_srcs_len} GREATER 0)
                cc_library(${TARGET}_part SRCS ${cpu_srcs} ${xpu_srcs} ${mkldnn_srcs} DEPS ${kernel_library_DEPS} ${kernel_deps})
                cc_library(${TARGET} SRCS ${common_srcs} ${selected_rows_srcs} DEPS ${TARGET}_part)
            endif()
        endif()
    # If there are only specific device srcs, build target using this rule.
    elseif (${cpu_srcs_len} GREATER 0 OR ${gpu_srcs_len} GREATER 0 OR ${xpu_srcs_len} GREATER 0 OR ${mkldnn_srcs_len} GREATER 0)
        if (WITH_GPU)
            if (${cpu_srcs_len} GREATER 0 OR ${gpu_srcs_len} GREATER 0 OR ${mkldnn_srcs_len} GREATER 0)
                nv_library(${TARGET} SRCS ${cpu_srcs} ${gpu_srcs} ${mkldnn_srcs} DEPS ${kernel_library_DEPS} ${kernel_deps})
            endif()
        elseif (WITH_ROCM)
            if (${cpu_srcs_len} GREATER 0 OR ${gpu_srcs_len} GREATER 0 OR ${mkldnn_srcs_len} GREATER 0)
                hip_library(${TARGET} SRCS ${cpu_srcs} ${gpu_srcs} ${mkldnn_srcs} DEPS ${kernel_library_DEPS} ${kernel_deps})
            endif()
        else()
            if (${cpu_srcs_len} GREATER 0 OR ${xpu_srcs_len} GREATER 0 OR ${mkldnn_srcs_len} GREATER 0)
                cc_library(${TARGET} SRCS ${cpu_srcs} ${xpu_srcs} ${mkldnn_srcs} DEPS ${kernel_library_DEPS} ${kernel_deps})
            endif()
        endif()
    # If the selected_rows_srcs depends on common_srcs, build target using this rule.
    elseif (${common_srcs_len} GREATER 0 AND ${selected_rows_srcs_len} GREATER 0)
        if (WITH_GPU)
            nv_library(${TARGET}_part SRCS ${common_srcs} DEPS ${kernel_library_DEPS} ${kernel_deps})
            nv_library(${TARGET} SRCS ${selected_rows_srcs} DEPS ${TARGET}_part)
        elseif (WITH_ROCM)
            hip_library(${TARGET}_part SRCS ${common_srcs} DEPS ${kernel_library_DEPS} ${kernel_deps})
            hip_library(${TARGET} SRCS ${selected_rows_srcs} DEPS ${TARGET}_part)
        else()
            cc_library(${TARGET}_part SRCS ${common_srcs} DEPS ${kernel_library_DEPS} ${kernel_deps})
            cc_library(${TARGET} SRCS ${selected_rows_srcs} DEPS ${TARGET}_part)
        endif()
    # If there are only common_srcs or selected_rows_srcs, build target using below rules.
    elseif (${common_srcs_len} GREATER 0)
        if (WITH_GPU)
            nv_library(${TARGET} SRCS ${common_srcs} DEPS ${kernel_library_DEPS} ${kernel_deps})
        elseif (WITH_ROCM)
            hip_library(${TARGET} SRCS ${common_srcs} DEPS ${kernel_library_DEPS} ${kernel_deps})
        else()
            cc_library(${TARGET} SRCS ${common_srcs} DEPS ${kernel_library_DEPS} ${kernel_deps})
        endif()
    elseif (${selected_rows_srcs_len} GREATER 0)
        if (WITH_GPU)
            nv_library(${TARGET} SRCS ${selected_rows_srcs} DEPS ${kernel_library_DEPS} ${kernel_deps})
        elseif (WITH_ROCM)
            hip_library(${TARGET} SRCS ${selected_rows_srcs} DEPS ${kernel_library_DEPS} ${kernel_deps})
        else()
            cc_library(${TARGET} SRCS ${selected_rows_srcs} DEPS ${kernel_library_DEPS} ${kernel_deps})
        endif()
    else()
         message(FATAL_ERROR "Cannot find any implementation for ${TARGET}")
    endif()

    if (${common_srcs_len} GREATER 0 OR ${cpu_srcs_len} GREATER 0 OR
        ${gpu_srcs_len} GREATER 0 OR ${xpu_srcs_len} GREATER 0  OR ${mkldnn_srcs_len} GREATER 0 OR 
        ${selected_rows_srcs_len} GREATER 0)
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
    if (${mkldnn_srcs_len} GREATER 0)
        kernel_declare(${mkldnn_srcs})
    endif()
    if (${selected_rows_srcs_len} GREATER 0)
        kernel_declare(${selected_rows_srcs})
    endif()
endfunction()

function(register_kernels)
    set(options "")
    set(oneValueArgs SUB_DIR)
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
                kernel_library(${target} DEPS ${register_kernels_DEPS} SUB_DIR ${register_kernels_SUB_DIR})
            else()
                kernel_library(${target} SUB_DIR ${register_kernels_SUB_DIR})
            endif()
        endif()
    endforeach()
endfunction()

function(append_op_util_declare TARGET)
    file(READ ${CMAKE_CURRENT_SOURCE_DIR}/${TARGET} target_content)
    string(REGEX MATCH "(PT_REGISTER_BASE_KERNEL_NAME|PT_REGISTER_ARG_MAPPING_FN)\\([ \t\r\n]*[a-z0-9_]*" util_registrar "${target_content}")
    string(REPLACE "PT_REGISTER_ARG_MAPPING_FN" "PT_DECLARE_ARG_MAPPING_FN" util_declare "${util_registrar}")
    string(REPLACE "PT_REGISTER_BASE_KERNEL_NAME" "PT_DECLARE_BASE_KERNEL_NAME" util_declare "${util_declare}")
    string(APPEND util_declare ");")
    file(APPEND ${op_utils_header} "${util_declare}")
endfunction()

function(register_op_utils TARGET_NAME)
    set(utils_srcs)
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs EXCLUDES DEPS)
    cmake_parse_arguments(register_op_utils "${options}" "${oneValueArgs}"
        "${multiValueArgs}" ${ARGN})

    file(GLOB SIGNATURES RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*_sig.cc")
    foreach(target ${SIGNATURES})
        append_op_util_declare(${target})
        list(APPEND utils_srcs ${CMAKE_CURRENT_SOURCE_DIR}/${target})
    endforeach()

    cc_library(${TARGET_NAME} SRCS ${utils_srcs} DEPS ${register_op_utils_DEPS})
endfunction()
