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

if(NOT WITH_XPU_KP)
    return()
endif()

if(NOT XPU_TOOLCHAIN)
  set(XPU_TOOLCHAIN /workspace/paddle/xpu-demo/XTDK)
  get_filename_component(XPU_TOOLCHAIN ${XPU_TOOLCHAIN} REALPATH)
endif()
if(NOT IS_DIRECTORY ${XPU_TOOLCHAIN})
  message(FATAL_ERROR "Directory ${XPU_TOOLCHAIN} not found!")
endif()
message(STATUS "Build with XPU_TOOLCHAIN=" ${XPU_TOOLCHAIN})
set(XPU_CLANG ${XPU_TOOLCHAIN}/bin/clang++)
message(STATUS "Build with XPU_CLANG=" ${XPU_CLANG})

# The host sysroot of XPU compiler is gcc-8.2 
if(NOT HOST_SYSROOT)
  set(HOST_SYSROOT /opt/compiler/gcc-8.2)
endif()

if(NOT IS_DIRECTORY ${HOST_SYSROOT})
  message(FATAL_ERROR "Directory ${HOST_SYSROOT} not found!")
endif()

if(NOT API_ARCH)
  set(API_ARCH x86_64-baidu-linux-gnu)
endif()

if(API_ARCH MATCHES "x86_64")
if(EXISTS ${HOST_SYSROOT}/bin/g++)
  set(HOST_CXX ${HOST_SYSROOT}/bin/g++)
  set(HOST_AR ${HOST_SYSROOT}/bin/ar)
else()
  set(HOST_CXX /usr/bin/g++)
  set(HOST_AR /usr/bin/ar)
endif()
else()
  set(HOST_CXX ${CMAKE_CXX_COMPILER})
  set(HOST_AR ${CMAKE_AR})
endif()

set(TOOLCHAIN_ARGS )

if(OPT_LEVEL)
  set(OPT_LEVEL ${OPT_LEVEL})
else()
  set(OPT_LEVEL "-O3")
endif()

message(STATUS "Build with API_ARCH=" ${API_ARCH})
message(STATUS "Build with TOOLCHAIN_ARGS=" ${TOOLCHAIN_ARGS})
message(STATUS "Build with HOST_SYSROOT=" ${HOST_SYSROOT})
message(STATUS "Build with HOST_CXX=" ${HOST_CXX})
message(STATUS "Build with HOST_AR=" ${HOST_AR})

macro(compile_kernel COMPILE_ARGS)
  set(options "")
  set(oneValueArgs "")
  set(multiValueArgs KERNEL DIRPATH XNAME DEVICE HOST XPU DEPENDS)
  cmake_parse_arguments(xpu_add_library "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  set(kernel_path ${xpu_add_library_DIRPATH})
  set(kernel_name ${xpu_add_library_XNAME})
  set(device_o_extra_flags ${xpu_add_library_DEVICE})
  set(host_o_extra_flags ${xpu_add_library_HOST})
  set(xpu_1_or_2 ${xpu_add_library_XPU})
  set(cc_depends ${xpu_add_library_DEPENDS})

  set(kernel_target ${kernel_name}_kernel)
  add_custom_target(${kernel_target}
    WORKING_DIRECTORY
      ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS
      kernel_build/${kernel_name}.host.o
      kernel_build/${kernel_name}.bin.o
    COMMENT
      ${kernel_target}
    VERBATIM
    )

  if(cc_depends)
    add_dependencies(${kernel_target} ${xpu_add_library_DEPENDS})
  endif()

  set(arg_device_o_extra_flags ${device_o_extra_flags})
  separate_arguments(arg_device_o_extra_flags)
  set(arg_host_o_extra_flags ${host_o_extra_flags})
  separate_arguments(arg_host_o_extra_flags)

  set(XTDK_DIR ${XPU_TOOLCHAIN})
  set(CXX_DIR ${HOST_SYSROOT})
  set(XPU_CXX_FLAGS  -Wno-error=pessimizing-move -Wno-error=constant-conversion -Wno-error=c++11-narrowing -Wno-error=shift-count-overflow -Wno-error=unused-local-typedef -Wno-error=deprecated-declarations -Wno-deprecated-declarations -std=c++14 -m64 -fPIC -fno-omit-frame-pointer  -Wall -Wno-inconsistent-missing-override -Wextra -Wnon-virtual-dtor -Wdelete-non-virtual-dtor -Wno-unused-parameter -Wno-unused-function  -Wno-error=unused-local-typedefs -Wno-error=ignored-attributes  -Wno-error=int-in-bool-context -Wno-error=parentheses -Wno-error=address -Wno-ignored-qualifiers -Wno-ignored-attributes -Wno-parentheses -DNDEBUG )

  #include path
  get_property(dirs DIRECTORY ${CMAKE_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
  set(XPU_CXX_INCLUDES "")
  foreach(dir IN LISTS dirs)
    list(APPEND XPU_CXX_INCLUDES "-I${dir}")
  endforeach()
  string(REPLACE ";" " " XPU_CXX_INCLUDES "${XPU_CXX_INCLUDES}" )
  separate_arguments(XPU_CXX_INCLUDES UNIX_COMMAND "${XPU_CXX_INCLUDES}")

  #related flags
  get_directory_property( DirDefs DIRECTORY ${CMAKE_SOURCE_DIR} COMPILE_DEFINITIONS )
  set(XPU_CXX_DEFINES "")
  foreach(def IN LISTS DirDefs)
    list(APPEND XPU_CXX_DEFINES "-D${def}")
  endforeach()
  string(REPLACE ";" " " XPU_CXX_DEFINES "${XPU_CXX_DEFINES}" )
  separate_arguments(XPU_CXX_DEFINES UNIX_COMMAND "${XPU_CXX_DEFINES}")

  add_custom_command(
    OUTPUT
      kernel_build/${kernel_name}.bin.o
    COMMAND
      ${CMAKE_COMMAND} -E make_directory kernel_build
    COMMAND
    ${XPU_CLANG} --sysroot=${CXX_DIR}  -std=c++11 -D_GLIBCXX_USE_CXX11_ABI=1 ${OPT_LEVEL} -fno-builtin -mcpu=xpu2  -fPIC ${XPU_CXX_DEFINES}  ${XPU_CXX_FLAGS}  ${XPU_CXX_INCLUDES} 
       -I.  -o kernel_build/${kernel_name}.bin.o.sec ${kernel_path}/${kernel_name}.xpu
        --xpu-device-only -c -v 
    COMMAND
      ${XTDK_DIR}/bin/xpu2-elfconv kernel_build/${kernel_name}.bin.o.sec  kernel_build/${kernel_name}.bin.o ${XPU_CLANG} --sysroot=${CXX_DIR}
    WORKING_DIRECTORY
      ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS
      ${xpu_add_library_DEPENDS}
    COMMENT
      kernel_build/${kernel_name}.bin.o
    VERBATIM
    )
    list(APPEND xpu_kernel_depends kernel_build/${kernel_name}.bin.o)

  add_custom_command(
    OUTPUT
      kernel_build/${kernel_name}.host.o
    COMMAND
      ${CMAKE_COMMAND} -E make_directory kernel_build
    COMMAND
    ${XPU_CLANG} --sysroot=${CXX_DIR}  -std=c++11 -D_GLIBCXX_USE_CXX11_ABI=1 ${OPT_LEVEL} -fno-builtin -mcpu=xpu2  -fPIC ${XPU_CXX_DEFINES}  ${XPU_CXX_FLAGS} ${XPU_CXX_INCLUDES} 
        -I.  -o kernel_build/${kernel_name}.host.o ${kernel_path}/${kernel_name}.xpu
        --xpu-host-only -c -v 
    WORKING_DIRECTORY
      ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS
      ${xpu_add_library_DEPENDS}
    COMMENT
      kernel_build/${kernel_name}.host.o
    VERBATIM
    )
    list(APPEND xpu_kernel_depends kernel_build/${kernel_name}.host.o)
endmacro()

###############################################################################
# XPU_ADD_LIBRARY
###############################################################################
macro(xpu_add_library TARGET_NAME)
    # Separate the sources from the options
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs STATIC DEPENDS)
    cmake_parse_arguments(xpu_add_library "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    set(xpu_srcs ${xpu_add_library_STATIC})
    set(xpu_target ${TARGET_NAME})
    set(cc_srcs_depends ${xpu_add_library_DEPENDS})
    
    file(GLOB_RECURSE xpu_srcs_lists ${xpu_srcs})
    list(LENGTH xpu_srcs_lists xpu_srcs_lists_num)

    set(XPU1_DEVICE_O_EXTRA_FLAGS " ")
    set(XPU1_HOST_O_EXTRA_FLAGS " ")

    # Distinguish .xpu file from other files
    foreach(cur_xpu_src IN LISTS xpu_srcs_lists)
      get_filename_component(language_type_name ${cur_xpu_src} EXT)
      if(${language_type_name} STREQUAL ".xpu")
        list(APPEND xpu_kernel_lists ${cur_xpu_src})
      else()
        list(APPEND cc_kernel_lists ${cur_xpu_src})
      endif()
    endforeach()

    # Ensure that there is only one xpu kernel
    list(LENGTH xpu_kernel_lists xpu_kernel_lists_num)
    list(LENGTH cc_srcs_depends cc_srcs_depends_num)

    if(${xpu_kernel_lists_num})
        foreach(xpu_kernel IN LISTS xpu_kernel_lists)
            get_filename_component(kernel_name ${xpu_kernel} NAME_WE)
            get_filename_component(kernel_dir ${xpu_kernel} DIRECTORY)
            set(kernel_rules ${kernel_dir}/${kernel_name}.rules)
            set(kernel_name ${kernel_name})
            compile_kernel( KERNEL ${xpu_kernel} DIRPATH ${kernel_dir} XNAME ${kernel_name} DEVICE ${XPU1_DEVICE_O_EXTRA_FLAGS} HOST ${XPU1_HOST_O_EXTRA_FLAGS} XPU "xpu2" DEPENDS ${cc_srcs_depends})
        endforeach()

        add_custom_target(${xpu_target}_src ALL
            WORKING_DIRECTORY
                ${CMAKE_CURRENT_BINARY_DIR}
            DEPENDS
                ${xpu_kernel_depends}
                ${CMAKE_CURRENT_BINARY_DIR}/lib${xpu_target}_xpu.a
            COMMENT
                ${xpu_target}_src
            VERBATIM
            )

        add_custom_command(
            OUTPUT
            ${CMAKE_CURRENT_BINARY_DIR}/lib${xpu_target}_xpu.a
            COMMAND
                ${HOST_AR} rcs ${CMAKE_CURRENT_BINARY_DIR}/lib${xpu_target}_xpu.a ${xpu_kernel_depends}
            WORKING_DIRECTORY
                ${CMAKE_CURRENT_BINARY_DIR}
            DEPENDS
                ${xpu_kernel_depends}
            COMMENT
                ${CMAKE_CURRENT_BINARY_DIR}/lib${xpu_target}_xpu.a
            VERBATIM
            ) 
        
        add_library(${xpu_target} STATIC ${cc_kernel_lists})
        add_dependencies(${xpu_target} ${xpu_target}_src)
        target_link_libraries(${TARGET_NAME} ${CMAKE_CURRENT_BINARY_DIR}/lib${xpu_target}_xpu.a)
    else()
        add_library(${xpu_target} STATIC ${cc_kernel_lists})
    endif()
endmacro()
