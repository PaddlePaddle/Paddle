if(NOT WITH_XPU2)
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
set(XPU_CLANG ${XPU_TOOLCHAIN}/bin/clang)
message(STATUS "Build with XPU_CLANG=" ${XPU_CLANG})

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
  set(OPT_LEVEL "-O2")
endif()

message(STATUS "Build with API_ARCH=" ${API_ARCH})
message(STATUS "Build with TOOLCHAIN_ARGS=" ${TOOLCHAIN_ARGS})
message(STATUS "Build with HOST_SYSROOT=" ${HOST_SYSROOT})
message(STATUS "Build with HOST_CXX=" ${HOST_CXX})
message(STATUS "Build with HOST_AR=" ${HOST_AR})

#macro(compile_kernel kernel_path kernel_name device_o_extra_flags host_o_extra_flags xpu_1_or_2 cc_depends)
macro(compile_kernel COMPILE_ARGS)
  set(options "")
  set(oneValueArgs "")
  set(multiValueArgs KERNEL XNAME DEVICE HOST XPU DEPENDS)
  cmake_parse_arguments(xpu_add_library "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  set(kernel_path ${xpu_add_library_KERNEL})
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
    message(STATUS "lxd_debug kernel dependencies: ${xpu_add_library_DEPENDS}")
    add_dependencies(${kernel_target} ${xpu_add_library_DEPENDS})
    #target_link_libraries(${kernel_target} ${xpu_add_library_DEPENDS})
  endif()

  # set(arg_rule ${rule})
  # separate_arguments(arg_rule)
  set(arg_device_o_extra_flags ${device_o_extra_flags})
  separate_arguments(arg_device_o_extra_flags)
  set(arg_host_o_extra_flags ${host_o_extra_flags})
  separate_arguments(arg_host_o_extra_flags)

  set(XTDK_DIR ${XPU_TOOLCHAIN})
  set(CXX_DIR ${HOST_SYSROOT})
  set(XPU_CXX_INCLUDES  -I/workspace/paddle/Paddle/build -I/workspace/paddle/Paddle/paddle/fluid/framework/io -I/workspace/paddle/Paddle/build/third_party/install/zlib/include -I/workspace/paddle/Paddle/build/third_party/install -I/workspace/paddle/Paddle/build/third_party/install/gflags/include -I/workspace/paddle/Paddle/build/third_party/install/glog/include -I/workspace/paddle/Paddle/build/third_party/boost/src/extern_boost -I/workspace/paddle/Paddle/build/third_party/eigen3/src/extern_eigen3 -I/workspace/paddle/Paddle/build/third_party/threadpool/src/extern_threadpool -I/workspace/paddle/Paddle/build/third_party/dlpack/src/extern_dlpack/include -I/workspace/paddle/Paddle/build/third_party/install/xxhash/include -I/workspace/paddle/Paddle/build/third_party/install/warpctc/include -I/workspace/paddle/Paddle/build/third_party/install/openblas/include -I/workspace/paddle/Paddle/build/third_party/install/protobuf/include -I/usr/include/python3.7m -I/usr/local/lib/python3.7/dist-packages/numpy/core/include -I/workspace/paddle/Paddle/build/third_party/pybind/src/extern_pybind/include -I/workspace/paddle/Paddle/build/third_party/install/gtest/include -I/workspace/paddle/Paddle/build/third_party/install/gloo/include -I/workspace/paddle/Paddle/build/third_party/install/xbyak/include -I/workspace/paddle/Paddle/build/third_party/install/xbyak/include/xbyak -I/workspace/paddle/Paddle/build/third_party/install/cryptopp/include -I/workspace/paddle/Paddle/build/third_party/pocketfft/src -I/workspace/paddle/Paddle)
  #set(XPU_CXX_FLAGS -Wno-error=deprecated-declarations -Wno-deprecated-declarations -std=c++14 -m64 -fPIC -fno-omit-frame-pointer -Werror -Wall -Wextra -Wnon-virtual-dtor -Wdelete-non-virtual-dtor -Wno-unused-parameter -Wno-unused-function -Wno-error=literal-suffix -Wno-error=unused-local-typedefs -Wno-error=ignored-attributes -Wno-error=terminate -Wno-error=int-in-bool-context -Wimplicit-fallthrough=0 -Wno-error=maybe-uninitialized -Wno-format-truncation -Wno-error=cast-function-type -Wno-error=parentheses -Wno-error=catch-value -Wno-error=nonnull-compare -Wno-error=address -Wno-ignored-qualifiers -Wno-ignored-attributes -Wno-parentheses -mavx -O3 -DNDEBUG )
  set(XPU_CXX_FLAGS  -Wno-error=constant-conversion -Wno-error=c++11-narrowing -Wno-error=shift-count-overflow -Wno-error=deprecated-declarations -Wno-deprecated-declarations -std=c++14 -m64 -fPIC -fno-omit-frame-pointer -Werror -Wall -Wno-inconsistent-missing-override -Wextra -Wnon-virtual-dtor -Wdelete-non-virtual-dtor -Wno-unused-parameter -Wno-unused-function  -Wno-error=unused-local-typedefs -Wno-error=ignored-attributes  -Wno-error=int-in-bool-context -Wno-error=parentheses -Wno-error=address -Wno-ignored-qualifiers -Wno-ignored-attributes -Wno-parentheses -O3 -DNDEBUG )
  #set(XPU_CXX_FLAGS -Wno-error=deprecated-declarations -Wno-deprecated-declarations -std=c++14 -m64 -fPIC -fno-omit-frame-pointer -Werror -Wall -Wextra -Wnon-virtual-dtor -Wdelete-non-virtual-dtor -Wno-unused-parameter -Wno-unused-function -Wno-error=literal-suffix -Wno-error=unused-local-typedefs -Wno-error=ignored-attributes -Wno-error=terminate -Wno-error=int-in-bool-context -Wimplicit-fallthrough=0 -Wno-error=maybe-uninitialized -Wno-format-truncation -Wno-error=cast-function-type -Wno-error=parentheses -Wno-error=catch-value -Wno-error=nonnull-compare -Wno-error=address -Wno-ignored-qualifiers -Wno-ignored-attributes -Wno-parentheses -O3 -DNDEBUG )
  set(XPU_CXX_DEFINES -DHPPL_STUB_FUNC -DPADDLE_DISABLE_PROFILER -DPADDLE_DLL_EXPORT -DPADDLE_USE_OPENBLAS -DPADDLE_USE_PTHREAD_BARRIER -DPADDLE_USE_PTHREAD_SPINLOCK -DPADDLE_VERSION=0.0.0 -DPADDLE_VERSION_INTEGER=0 -DPADDLE_WITH_AVX -DPADDLE_WITH_CRYPTO -DPADDLE_WITH_POCKETFFT -DPADDLE_WITH_SSE3 -DPADDLE_WITH_TESTING -DPADDLE_WITH_XBYAK -DPADDLE_WITH_XPU2 -DXBYAK64 -DXBYAK_NO_OP_NAMES)

  add_custom_command(
    OUTPUT
      kernel_build/${kernel_name}.bin.o
    COMMAND
      ${CMAKE_COMMAND} -E make_directory kernel_build
    COMMAND
    # TODO(liuxiandong) xpu->kps -I${XTDK_DIR}/include -std=c++11
    ${XPU_CLANG} --sysroot=${CXX_DIR}   -O2 -fno-builtin -g -mcpu=xpu2  -fPIC ${XPU_CXX_DEFINES}  ${XPU_CXX_FLAGS}  ${XPU_CXX_INCLUDES} 
        -I${XTDK_DIR}/include -I.  -o kernel_build/${kernel_name}.bin.o.sec /workspace/paddle/Paddle/paddle/fluid/operators/elementwise/${kernel_name}.xpu
        --xpu-device-only -c -v 
    COMMAND
      ${XTDK_DIR}/bin/xpu2-elfconv kernel_build/${kernel_name}.bin.o.sec  kernel_build/${kernel_name}.bin.o ${XPU_CLANG} --sysroot=${CXX_DIR}
    # COMMAND
    #   ${CMAKE_COMMAND} -E cmake_depends "Unix Makefiles" ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR}
    #     ${CMAKE_BINARY_DIR} ${CMAKE_BINARY_DIR}
    #     ${CMAKE_BINARY_DIR}/CMakeFiles/${kernel_target}.dir/DependInfo.cmake --color=$(COLOR)
    WORKING_DIRECTORY
      ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS
      ${xpu_add_library_DEPENDS}
    COMMENT
      kernel_build/${kernel_name}.bin.o
    VERBATIM
    )
    # TODO attention here
    #set(xpu_kernel_depends ${kernel_name}_depends)
    list(APPEND xpu_kernel_depends kernel_build/${kernel_name}.bin.o)

  add_custom_command(
    OUTPUT
      kernel_build/${kernel_name}.host.o
    COMMAND
      ${CMAKE_COMMAND} -E make_directory kernel_build
    COMMAND
    # TODO(liuxiandong) xpu->kps -I${XTDK_DIR}/include -std=c++11
    ${XPU_CLANG} --sysroot=${CXX_DIR}   -O2 -fno-builtin -g -mcpu=xpu2  -fPIC ${XPU_CXX_DEFINES}  ${XPU_CXX_FLAGS} ${XPU_CXX_INCLUDES} 
        -I${XTDK_DIR}/include -I.  -o kernel_build/${kernel_name}.host.o /workspace/paddle/Paddle/paddle/fluid/operators/elementwise/${kernel_name}.xpu
        --xpu-host-only -c -v 
    # COMMAND
    #   ${CMAKE_COMMAND} -E cmake_depends "Unix Makefiles" ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR}
    #     ${CMAKE_BINARY_DIR} ${CMAKE_BINARY_DIR}
    #     ${CMAKE_BINARY_DIR}/CMakeFiles/${kernel_target}.dir/DependInfo.cmake --color=$(COLOR)
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

# macro(__compile_kernel_with_rules kernel_path kernel_name rules_path device_o_extra_flags host_o_extra_flags xpu_1_or_2 cc_depends)
#   file(STRINGS ${rules_path} rules)
#   foreach(rule IN LISTS rules)
#     message(STATUS "  Instantiate with '${rule}'")
#     execute_process(
#       COMMAND
#         bash "-c" "echo -n ${rule} | md5sum | cut -c1-6"
#       OUTPUT_VARIABLE
#         rule_md5
#       OUTPUT_STRIP_TRAILING_WHITESPACE
#       )

#     set(kernel_name_md5 ${kernel_name}_${rule_md5})
#     compile_kernel(${kernel_path} ${kernel_name_md5} ${rule} ${device_o_extra_flags} ${host_o_extra_flags} ${xpu_1_or_2} ${cc_depends})
#   endforeach()
# endmacro()

# macro(compile_kernel_with_rules kernel_path kernel_name rules_path device_o_extra_flags host_o_extra_flags xpu_1_or_2 cc_depends)
#   # XXX: reconfigure if file |rules_path| was modified
#   set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${rules_path})
#   __compile_kernel_with_rules(${kernel_path} ${kernel_name} ${rules_path} ${device_o_extra_flags} ${host_o_extra_flags} ${xpu_1_or_2} ${cc_depends})
# endmacro()

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
    #message(STATUS "lxd_debug: ${xpu_add_library_DEPENDS}---------------------------------")
    
    file(GLOB_RECURSE xpu_srcs_lists ${xpu_srcs})
    list(LENGTH xpu_srcs_lists xpu_srcs_lists_num)

    set(XPU1_DEVICE_O_EXTRA_FLAGS " ")
    set(XPU1_HOST_O_EXTRA_FLAGS " ")

    # Distinguish .xpu file from other files
    foreach(cur_xpu_src IN LISTS xpu_srcs_lists)
      get_filename_component(language_type_name ${cur_xpu_src} EXT)
      # TODO(liuxiandong) xpu->kps
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
            message(STATUS "Process ${xpu_kernel}")
            get_filename_component(kernel_name ${xpu_kernel} NAME_WE)
            get_filename_component(kernel_dir ${xpu_kernel} DIRECTORY)
            #TODO(liuxiandong set default rules)
            set(kernel_rules ${kernel_dir}/${kernel_name}.rules)
            set(kernel_name ${kernel_name})
            # if(EXISTS ${kernel_rules})
            #     # compile_kernel_with_rules(${xpu_kernel} ${kernel_name} ${kernel_rules}
            #     #     ${XPU1_DEVICE_O_EXTRA_FLAGS} ${XPU1_HOST_O_EXTRA_FLAGS} "xpu2" ${cc_srcs_depends})
            # else()
            message(STATUS "lxd_debug: ${cc_srcs_depends}>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            compile_kernel(KERNEL ${xpu_kernel} XNAME ${kernel_name} DEVICE ${XPU1_DEVICE_O_EXTRA_FLAGS} HOST ${XPU1_HOST_O_EXTRA_FLAGS} XPU "xpu2" DEPENDS ${cc_srcs_depends})
            # endif()
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
        
        # if(${cc_srcs_depends_num})
        #   add_dependencies(${xpu_target}_kernel ${cc_srcs_depends})
        # endif()
        add_library(${xpu_target} STATIC ${cc_kernel_lists})
        target_link_libraries(${TARGET_NAME} ${CMAKE_CURRENT_BINARY_DIR}/lib${xpu_target}_xpu.a)
    else()
        add_library(${xpu_target} STATIC ${cc_kernel_lists})
    endif()
endmacro()

# XPU2 PATH
if(NOT DEFINED ENV{XPU2_PATH})
    set(XPU2_PATH "/workspace/paddle/xpu-demo/XTDK" CACHE PATH "Path to which XPU2 has been installed")
    set(XPU_CLANG_PATH ${XPU2_PATH}/bin/clang CACHE PATH "Path to which XPU2 CLANG has been installed")
else()
    set(XPU2_PATH $ENV{XPU2_PATH} CACHE PATH "Path to which ROCm has been installed")
    set(XPU_CLANG_PATH ${XPU2_PATH}/bin/clang CACHE PATH "Path to which XPU2 CLANG has been installed")
endif()
set(CMAKE_MODULE_PATH "${XPU2_CLANG_PATH}/cmake" ${CMAKE_MODULE_PATH})

# define XPU_CXX_FLAGS
list(APPEND XPU_CFLAGS -fPIC)
list(APPEND XPU_CFLAGS --sysroot = /opt/compiler/gcc-8.2)
list(APPEND XPU_CFLAGS -std=c++11)
list(APPEND XPU_CFLAGS -O2)
list(APPEND XPU_CFLAGS -g)
list(APPEND XPU_CFLAGS -mcpu=xpu2)
list(APPEND XPU_CFLAGS --target=x86_64-linux-gnu)
list(APPEND XPU_CFLAGS -v)
list(APPEND XPU_CFLAGS --dyld-prefix=/opt/compiler/gcc-8.2)
list(APPEND XPU_CFLAGS -fno-builtin)
list(APPEND XPU_CFLAGS -Wno-dev)

set(XPU_XPUCC_FLAGS ${XPU_CFLAGS})

# set HIP link libs
set(xpuapi_library_name xpuapi)
message(STATUS "XPU API library name: ${xpuapi_library_name}")
# link in the generic.cmake
find_library(XPU2_CLANG_API_LIB ${xpuapi_library_name} HINTS ${XPU2_PATH}/shlib)
message(STATUS "XPU2_CLANG_API_LIB: ${XPU2_CLANG_API_LIB}")

set(xpurt_library_name xpurt)
message(STATUS "XPU RT library name: ${xpurt_library_name}")
# link in the generic.cmake
find_library(XPU2_CLANG_RT_LIB ${xpurt_library_name} HINTS ${XPU2_PATH}/runtime/shlib)
message(STATUS "XPU2_CLANG_RT_LIB: ${XPU2_CLANG_RT_LIB}")

# # Ensure that xpu/api.h can be included without dependency errors.
# file(GENERATE OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/.xpu_headers_dummy.cc CONTENT "")
# add_library(xpu_headers_dummy STATIC ${CMAKE_CURRENT_BINARY_DIR}/.xpu_headers_dummy.cc)
# add_dependencies(xpu_headers_dummy extern_xpu)
# link_libraries(xpu_headers_dummy)