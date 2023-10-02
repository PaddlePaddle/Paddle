cmake_minimum_required(VERSION 3.10)

project(xpuplugin LANGUAGES CXX)

if(NOT DEFINED BUILD_STANDALONE)
  if(NOT DEFINED XPU_INC_DIR)
    message(
      FATAL_ERROR
        "XPU_INC_DIR not set, or directory ${XPU_INC_DIR} not found, please compile with PaddlePaddle."
    )
  endif()
  if(NOT DEFINED XPU_LIB_DIR)
    message(
      FATAL_ERROR
        "XPU_LIB_DIR not set, or directory ${XPU_LIB_DIR} not found, please compile with PaddlePaddle."
    )
  endif()
  set(XDNN_INC_DIR ${XPU_INC_DIR})
  set(XDNN_LIB_DIR ${XPU_LIB_DIR})
  set(XRE_INC_DIR ${XPU_INC_DIR})
  set(XRE_LIB_DIR ${XPU_LIB_DIR})
  set(XPU_DEPS xpulib) # Depends cmake/external/xpu.cmake
else()
  if(NOT DEFINED XDNN_PATH)
    set(XDNN_PATH $ENV{XDNN_PATH})
  endif()
  if(NOT DEFINED XRE_PATH)
    set(XRE_PATH $ENV{XRE_PATH})
  endif()
  if(NOT IS_DIRECTORY ${XDNN_PATH})
    message(
      FATAL_ERROR
        "XDNN_PATH not set, or directory ${XDNN_PATH} not found, please export XDNN_PATH=<path_to_xdnn>."
    )
  endif()
  if(NOT IS_DIRECTORY ${XRE_PATH})
    message(
      FATAL_ERROR
        "XRE_PATH not set, or directory ${XRE_PATH} not found, please export XRE_PATH=<path_to_xre>."
    )
  endif()
  set(XDNN_INC_DIR ${XDNN_PATH}/include)
  set(XDNN_LIB_DIR ${XDNN_PATH}/so)
  set(XRE_INC_DIR ${XRE_PATH}/include)
  set(XRE_LIB_DIR ${XRE_PATH}/so)
endif()

if(NOT DEFINED CLANG_PATH)
  set(CLANG_PATH $ENV{CLANG_PATH})
endif()
if(NOT IS_DIRECTORY ${CLANG_PATH})
  message(
    FATAL_ERROR
      "Directory ${CLANG_PATH} not found, please export CLANG_PATH=<path_to_xtdk>."
  )
endif()

message(STATUS "Build with CLANG_PATH=" ${CLANG_PATH})
set(XPU_CLANG ${CLANG_PATH}/bin/clang++)
message(STATUS "Build with XPU_CLANG=" ${XPU_CLANG})

if(NOT DEFINED HOST_SYSROOT)
  set(HOST_SYSROOT $ENV{HOST_SYSROOT})
endif()
if(HOST_SYSROOT)
  if(NOT IS_DIRECTORY ${HOST_SYSROOT})
    message(
      FATAL_ERROR
        "Directory ${HOST_SYSROOT} not found, please export HOST_SYSROOT=<path_to_gcc>."
    )
  endif()
endif()

if(NOT DEFINED HOST_ARCH)
  set(HOST_ARCH $ENV{HOST_ARCH})
endif()
if(NOT HOST_ARCH)
  set(HOST_ARCH x86_64-baidu-linux-gnu)
endif()

if(NOT DEFINED TARGET_ARCH)
  set(TARGET_ARCH $ENV{TARGET_ARCH})
endif()
if(NOT TARGET_ARCH)
  set(TARGET_ARCH x86_64-baidu-linux-gnu)
endif()

if(NOT DEFINED TOOLCHAIN_ARGS)
  set(TOOLCHAIN_ARGS $ENV{TOOLCHAIN_ARGS})
endif()
if(HOST_ARCH MATCHES "x86_64")
  if(TARGET_ARCH MATCHES "x86_64")
    if(EXISTS ${HOST_SYSROOT}/bin/g++)
      set(HOST_CXX ${HOST_SYSROOT}/bin/g++)
      set(HOST_AR ${HOST_SYSROOT}/bin/ar)
      if(NOT EXISTS ${HOST_AR})
        # try gcc-ar
        set(HOST_AR ${HOST_SYSROOT}/bin/gcc-ar)
      endif()
    else()
      set(HOST_CXX /usr/bin/g++)
      set(HOST_AR /usr/bin/ar)
    endif()
  endif()
  if(TARGET_ARCH MATCHES "aarch64")
    set(TOOLCHAIN_ARGS "${TOOLCHAIN_ARGS} --gcc-toolchain=${HOST_SYSROOT}")
    set(HOST_SYSROOT ${HOST_SYSROOT}/aarch64-linux-gnu/libc)
    set(HOST_CXX ${CMAKE_CXX_COMPILER})
    set(HOST_AR ${CMAKE_AR})
  endif()
endif()
if(HOST_ARCH MATCHES "aarch64")
  if(TARGET_ARCH MATCHES "aarch64")
    if(EXISTS ${HOST_SYSROOT}/bin/g++)
      set(HOST_CXX ${HOST_SYSROOT}/bin/g++)
      set(HOST_AR ${HOST_SYSROOT}/bin/ar)
    else()
      set(HOST_CXX /usr/bin/g++)
      set(HOST_AR /usr/bin/ar)
    endif()
  endif()
endif()

set(OPT_LEVEL "-O2")
message(STATUS "Build with TARGET_ARCH=" ${TARGET_ARCH})
message(STATUS "Build with TOOLCHAIN_ARGS=" ${TOOLCHAIN_ARGS})
message(STATUS "Build with HOST_SYSROOT=" ${HOST_SYSROOT})
message(STATUS "Build with HOST_CXX=" ${HOST_CXX})
message(STATUS "Build with HOST_AR=" ${HOST_AR})

separate_arguments(TOOLCHAIN_ARGS)
# compile xpu kernel macro function
macro(
  compile_kernel
  kernel_path
  kernel_name
  xpu_n
  rule
  device_o_extra_flags
  host_o_extra_flags
  xpu_n_macro)
  set(arg_rule ${rule})
  separate_arguments(arg_rule)
  set(arg_device_o_extra_flags ${device_o_extra_flags})
  separate_arguments(arg_device_o_extra_flags)
  set(arg_host_o_extra_flags ${host_o_extra_flags})
  separate_arguments(arg_host_o_extra_flags)

  add_custom_command(
    OUTPUT ${kernel_name}.device.bin.o ${kernel_name}.o
    COMMAND
      ${XPU_CLANG} -std=c++11 ${OPT_LEVEL} ${arg_device_o_extra_flags} -c
      ${kernel_path} -D ${xpu_n_macro} --target=${TARGET_ARCH} ${HOST_XPU_FLAGS}
      --basename ${kernel_name} -fno-builtin --xpu-arch=${xpu_n} -fPIC
      -Wno-int-to-void-pointer-cast -Wno-int-to-pointer-cast -Werror -mllvm
      --xpu-inline-cost -mllvm --xpu-inline-hot-call -I${XDNN_INC_DIR}
      -I${CMAKE_CURRENT_SOURCE_DIR}/include -I${CMAKE_CURRENT_SOURCE_DIR}/src
      -I${CMAKE_CURRENT_SOURCE_DIR}/src/kernel
      -I${CMAKE_CURRENT_SOURCE_DIR}/src/kernel/include ${arg_rule}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${kernel_path}
    COMMENT ${kernel_name}.device.bin.o ${kernel_name}.o
    VERBATIM)

  list(APPEND xpuplugin_kernels_depends ${kernel_name}.device.bin.o
       ${kernel_name}.o)
endmacro()

macro(
  __compile_kernel_with_rules
  kernel_path
  kernel_name
  xpu_n
  rules_path
  device_o_extra_flags
  host_o_extra_flags
  xpu_n_macro)
  file(STRINGS ${rules_path} rules)

  foreach(rule IN LISTS rules)
    message(STATUS "  Instantiate with '${rule}'")
    execute_process(
      COMMAND bash "-c" "echo -n ${rule} | md5sum | cut -c1-6"
      OUTPUT_VARIABLE rule_md5
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    set(kernel_name_md5 ${kernel_name}_${rule_md5})
    compile_kernel(
      ${kernel_path}
      ${kernel_name_md5}
      ${xpu_n}
      ${rule}
      ${device_o_extra_flags}
      ${host_o_extra_flags}
      ${xpu_n_macro})
  endforeach()
endmacro()

macro(
  compile_kernel_with_rules
  kernel_path
  kernel_name
  xpu_n
  rules_path
  device_o_extra_flags
  host_o_extra_flags
  xpu_n_macro)
  # reconfigure if file |rules_path| was modified
  set_property(
    DIRECTORY
    APPEND
    PROPERTY CMAKE_CONFIGURE_DEPENDS ${rules_path})
  __compile_kernel_with_rules(
    ${kernel_path}
    ${kernel_name}
    ${xpu_n}
    ${rules_path}
    ${device_o_extra_flags}
    ${host_o_extra_flags}
    ${xpu_n_macro})
endmacro()

macro(search_and_compile_kernel xpu_n)
  if(${xpu_n} STREQUAL "xpu1")
    set(XPU_DEVICE_O_EXTRA_FLAGS " ")
    set(XPU_HOST_O_EXTRA_FLAGS " ")
    set(XPU_KERNEL_PATH "src/kernel/cpp/*.xpu")
    set(xpu_n_macro "__XPU1__")
  elseif(${xpu_n} STREQUAL "xpu2")
    set(XPU_DEVICE_O_EXTRA_FLAGS "--xpu-arch=xpu2")
    set(XPU_HOST_O_EXTRA_FLAGS "--xpu-arch=xpu2")
    set(XPU_KERNEL_PATH "src/kernel/kunlun2cpp/*.xpu")
    set(xpu_n_macro "__XPU2__")
  elseif(${xpu_n} STREQUAL "xpu3")
    set(XPU_DEVICE_O_EXTRA_FLAGS "--xpu-arch=xpu3")
    set(XPU_HOST_O_EXTRA_FLAGS "--xpu-arch=xpu3")
    set(XPU_KERNEL_PATH "src/kernel/kunlun3cpp/*.xpu")
    set(xpu_n_macro "__XPU3__")
  else()
    message(FATAL_ERROR "Are you sure? ${xpu_n}")
  endif()
  file(GLOB_RECURSE xpu_kernels ${XPU_KERNEL_PATH})
  list(LENGTH xpu_kernels xpu_kernels_num)
  message(STATUS "Found ${xpu_kernels_num} ${xpu_n} kernels")

  foreach(xpu_kernel IN LISTS xpu_kernels)
    message(STATUS "Process ${xpu_kernel}")
    get_filename_component(kernel_name ${xpu_kernel} NAME_WE)
    get_filename_component(kernel_dir ${xpu_kernel} DIRECTORY)
    set(kernel_rules ${kernel_dir}/${kernel_name}.rules)
    set(kernel_name ${xpu_n}_${kernel_name})
    if(EXISTS ${kernel_rules})
      compile_kernel_with_rules(
        ${xpu_kernel}
        ${kernel_name}
        ${xpu_n}
        ${kernel_rules}
        ${XPU_DEVICE_O_EXTRA_FLAGS}
        ${XPU_HOST_O_EXTRA_FLAGS}
        ${xpu_n_macro})
    else()
      compile_kernel(
        ${xpu_kernel}
        ${kernel_name}
        ${xpu_n}
        " "
        ${XPU_DEVICE_O_EXTRA_FLAGS}
        ${XPU_HOST_O_EXTRA_FLAGS}
        ${xpu_n_macro})
    endif()
  endforeach()
endmacro()

# compile xpu kernels
search_and_compile_kernel("xpu1")
search_and_compile_kernel("xpu2")
search_and_compile_kernel("xpu3")

# compile xpu wrappers
file(GLOB_RECURSE xpu_wrappers src/wrapper/*.cpp)
list(LENGTH xpu_wrappers xpu_wrappers_num)
message(STATUS "Found ${xpu_wrappers_num} XPU wrappers")

foreach(xpu_wrapper IN LISTS xpu_wrappers)
  message(STATUS "Process ${xpu_wrapper}")
  get_filename_component(wrapper_name ${xpu_wrapper} NAME_WE)
  set(wrapper_target ${wrapper_name}_wrapper)

  add_custom_target(
    ${wrapper_target}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS wrapper_build/${wrapper_name}.wrapper.d
            wrapper_build/${wrapper_name}.wrapper.o
    COMMENT ${wrapper_target}
    VERBATIM)

  add_custom_command(
    OUTPUT wrapper_build/${wrapper_name}.wrapper.d
    COMMAND ${CMAKE_COMMAND} -E make_directory wrapper_build
    COMMAND
      ${XPU_CLANG} -M -MQ wrapper_build/${wrapper_name}.wrapper.o -MF
      wrapper_build/${wrapper_name}.wrapper.d -std=c++11 -x xpu -c
      ${xpu_wrapper} -I${XDNN_INC_DIR} -I${XRE_INC_DIR}
      -I${CMAKE_CURRENT_SOURCE_DIR}/include -I${CMAKE_CURRENT_SOURCE_DIR}/src
      -I${CMAKE_CURRENT_SOURCE_DIR}/src/wrapper -D_GNU_SOURCE
      -D__STDC_LIMIT_MACROS -DNDEBUG ${TOOLCHAIN_ARGS} --target=${TARGET_ARCH}
      -fPIC -Werror -Wreorder -fvisibility=hidden --xpu-host-only
      ${XPU_MF_FLAGS}
    COMMAND
      ${CMAKE_COMMAND} -E cmake_depends "Unix Makefiles" ${CMAKE_SOURCE_DIR}
      ${CMAKE_SOURCE_DIR} ${CMAKE_BINARY_DIR} ${CMAKE_BINARY_DIR}
      ${CMAKE_BINARY_DIR}/CMakeFiles/${wrapper_target}.dir/DependInfo.cmake
      --color=$(COLOR)
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${xpu_wrapper} ${XPU_DEPS}
    COMMENT wrapper_build/${wrapper_name}.wrapper.d
    VERBATIM)

  add_custom_command(
    OUTPUT wrapper_build/${wrapper_name}.wrapper.o
    COMMAND ${CMAKE_COMMAND} -E make_directory wrapper_build
    COMMAND
      ${XPU_CLANG} -std=c++11 ${EXTRA_FLAGS} ${OPT_LEVEL} -x xpu -c
      ${xpu_wrapper} -o wrapper_build/${wrapper_name}.wrapper.o
      -I${XDNN_INC_DIR} -I${XRE_INC_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/include
      -I${CMAKE_CURRENT_SOURCE_DIR}/src
      -I${CMAKE_CURRENT_SOURCE_DIR}/src/wrapper -D_GNU_SOURCE
      -D__STDC_LIMIT_MACROS -DNDEBUG ${TOOLCHAIN_ARGS} --target=${TARGET_ARCH}
      -fPIC -Wunused-variable -Werror -Wreorder -fvisibility=hidden
      --xpu-host-only ${HOST_XPU_FLAGS}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS wrapper_build/${wrapper_name}.wrapper.d
    COMMENT wrapper_build/${wrapper_name}.wrapper.o
    VERBATIM)
  list(APPEND xpuplugin_wrapper_depends wrapper_build/${wrapper_name}.wrapper.o)
endforeach()

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/libxpuplugin.a
  COMMAND ${HOST_AR} rcs ${CMAKE_CURRENT_BINARY_DIR}/libxpuplugin.a
          ${xpuplugin_kernels_depends} ${xpuplugin_wrapper_depends}
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${xpuplugin_kernels_depends} ${xpuplugin_wrapper_depends}
  COMMENT ${CMAKE_CURRENT_BINARY_DIR}/libxpuplugin.a
  VERBATIM)

add_custom_target(
  xpuplugin_a
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${xpuplugin_kernels_depends} ${xpuplugin_wrapper_depends}
          ${CMAKE_CURRENT_BINARY_DIR}/libxpuplugin.a
  COMMENT xpuplugin_a
  VERBATIM)

add_custom_target(
  xpuplugin_so ALL
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS xpuplugin_a ${CMAKE_CURRENT_BINARY_DIR}/libxpuplugin.so
  COMMENT xpuplugin_so)

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/libxpuplugin.so
  COMMAND
    ${HOST_CXX} -shared -o ${CMAKE_CURRENT_BINARY_DIR}/libxpuplugin.so -Xlinker
    \"-\(\" -Wl,--whole-archive ${CMAKE_CURRENT_BINARY_DIR}/libxpuplugin.a
    -Wl,--no-whole-archive -L${XDNN_LIB_DIR} -L${XRE_LIB_DIR} -lxpurt -lxpuapi
    -Wl,--no-undefined -Wl,-soname,libxpuplugin.so -lstdc++ -ldl -lm -lpthread
    -specs=${CMAKE_CURRENT_SOURCE_DIR}/src/linker.specs -Xlinker \"-\)\"\;
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libxpuplugin.a
  COMMENT ${CMAKE_CURRENT_BINARY_DIR}/libxpuplugin.so)

if(NOT DEFINED BUILD_STANDALONE)
  add_library(xpuplugin STATIC IMPORTED GLOBAL)
  add_dependencies(xpuplugin xpuplugin_a)
  set_target_properties(
    xpuplugin PROPERTIES IMPORTED_LOCATION
                         ${CMAKE_CURRENT_BINARY_DIR}/libxpuplugin.a)
endif()
