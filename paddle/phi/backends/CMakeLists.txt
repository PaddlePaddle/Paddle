add_subdirectory(dynload)
add_subdirectory(gpu)

set(BACKENDS_SRCS all_context.cc cpu/cpu_context.cc cpu/cpu_info.cc)
set(BACKENDS_DEPS enforce place flags eigen3 phi_device_context)
if(WITH_XBYAK)
  list(APPEND BACKENDS_DEPS xbyak)
endif()

function(get_compiled_arch out_variable)
  # List of arch names
  set(archs_names
      "Kepler"
      "Maxwell"
      "Pascal"
      "Volta"
      "Turing"
      "Ampere"
      "All"
      "Manual")
  set(archs_name_default "Auto")
  list(APPEND archs_names "Auto")

  # set CUDA_ARCH_NAME strings (so it will be seen as dropbox in CMake-Gui)
  set(CUDA_ARCH_NAME
      ${archs_name_default}
      CACHE STRING "Select target NVIDIA GPU achitecture.")
  set_property(CACHE CUDA_ARCH_NAME PROPERTY STRINGS "" ${archs_names})
  mark_as_advanced(CUDA_ARCH_NAME)

  # verify CUDA_ARCH_NAME value
  if(NOT ";${archs_names};" MATCHES ";${CUDA_ARCH_NAME};")
    string(REPLACE ";" ", " archs_names "${archs_names}")
    message(
      FATAL_ERROR "Only ${archs_names} architectures names are supported.")
  endif()

  if(${CUDA_ARCH_NAME} STREQUAL "Manual")
    set(CUDA_ARCH_BIN
        ${paddle_known_gpu_archs}
        CACHE
          STRING
          "Specify 'real' GPU architectures to build binaries for, BIN(PTX) format is supported"
    )
    mark_as_advanced(CUDA_ARCH_BIN)
  else()
    unset(CUDA_ARCH_BIN CACHE)
  endif()

  if(${CUDA_ARCH_NAME} STREQUAL "Kepler")
    set(cuda_arch_bin "30 35")
  elseif(${CUDA_ARCH_NAME} STREQUAL "Maxwell")
    if(WITH_NV_JETSON)
      set(cuda_arch_bin "53")
    else()
      set(cuda_arch_bin "50")
    endif()
  elseif(${CUDA_ARCH_NAME} STREQUAL "Pascal")
    if(WITH_NV_JETSON)
      set(cuda_arch_bin "62")
    else()
      set(cuda_arch_bin "60 61")
    endif()
  elseif(${CUDA_ARCH_NAME} STREQUAL "Volta")
    if(WITH_NV_JETSON)
      set(cuda_arch_bin "72")
    else()
      set(cuda_arch_bin "70")
    endif()
  elseif(${CUDA_ARCH_NAME} STREQUAL "Turing")
    set(cuda_arch_bin "75")
  elseif(${CUDA_ARCH_NAME} STREQUAL "Ampere")
    if(WITH_NV_JETSON)
      set(cuda_arch_bin "87")
    else()
      if(${CMAKE_CUDA_COMPILER_VERSION} LESS 11.1) # CUDA 11.0
        set(cuda_arch_bin "80")
      elseif(${CMAKE_CUDA_COMPILER_VERSION} LESS 12.0) # CUDA 11.1+
        set(cuda_arch_bin "80 86")
      endif()
    endif()
    message("cuda arch ${cuda_arch_bin}")

  elseif(${CUDA_ARCH_NAME} STREQUAL "All")
    set(cuda_arch_bin ${paddle_known_gpu_archs})
  elseif(${CUDA_ARCH_NAME} STREQUAL "Auto")
    message(
      STATUS
        "WARNING: This is just a warning for publishing release.
      You are building GPU version without supporting different architectures.
      So the wheel package may fail on other GPU architectures.
      You can add -DCUDA_ARCH_NAME=All in cmake command
      to get a full wheel package to resolve this warning.
      While, this version will still work on local GPU architecture.")
    detect_installed_gpus(cuda_arch_bin)
  else() # (${CUDA_ARCH_NAME} STREQUAL "Manual")
    set(cuda_arch_bin ${CUDA_ARCH_BIN})
  endif()

  # remove dots and convert to lists
  string(REGEX REPLACE "\\." "" cuda_arch_bin "${cuda_arch_bin}")
  string(REGEX MATCHALL "[0-9()]+" cuda_arch_bin "${cuda_arch_bin}")
  list(REMOVE_DUPLICATES cuda_arch_bin)
  # add comma to each cuda arch
  string(JOIN "," cuda_real_archs ${cuda_arch_bin})
  set(${out_variable}
      ${cuda_real_archs}
      PARENT_SCOPE)
endfunction()

if(WITH_GPU OR WITH_ROCM)
  list(APPEND BACKENDS_SRCS gpu/gpu_context.cc gpu/gpu_info.cc
       gpu/gpu_resources.cc)
  if(WITH_GPU)
    list(APPEND BACKENDS_SRCS gpu/cuda/cuda_info.cc gpu/cuda/cuda_graph.cc)
    get_compiled_arch(CUDA_REAL_ARCH)
    set_source_files_properties(
      gpu/gpu_resources.cc PROPERTIES COMPILE_FLAGS
                                      "-DCUDA_REAL_ARCHS=\"${CUDA_REAL_ARCH}\"")

  endif()
  if(WITH_ROCM)
    list(APPEND BACKENDS_SRCS gpu/rocm/rocm_info.cc)
  endif()
  list(APPEND BACKENDS_DEPS phi_dynload_cuda)
endif()

if(WITH_XPU)
  list(APPEND BACKENDS_SRCS xpu/xpu_context.cc xpu/xpu_info.cc)
  list(APPEND BACKENDS_SRCS xpu/xpu_op_list.cc xpu/xpu1_op_list.cc
       xpu/xpu2_op_list.cc)
endif()

if(WITH_MKLDNN)
  list(APPEND BACKENDS_SRCS onednn/onednn_context.cc)
  list(APPEND BACKENDS_SRCS onednn/axpy_handler.cc)
  list(APPEND BACKENDS_DEPS mkldnn)
endif()

if(WITH_CUSTOM_DEVICE)
  list(
    APPEND
    BACKENDS_SRCS
    callback_manager.cc
    device_guard.cc
    stream.cc
    event.cc
    device_base.cc
    device_manager.cc
    custom/custom_context.cc
    custom/custom_device.cc)
endif()

add_library(phi_backends "${BACKENDS_SRCS}")
target_link_libraries(phi_backends ${BACKENDS_DEPS})
add_dependencies(phi_backends eigen3)

# for inference library
get_property(phi_modules GLOBAL PROPERTY PHI_MODULES)
set(phi_modules ${phi_modules} phi_backends)
set_property(GLOBAL PROPERTY PHI_MODULES "${phi_modules}")

if(WITH_CUSTOM_DEVICE)
  cc_test(
    custom_device_test
    SRCS custom/custom_device_test.cc
    DEPS phi_backends phi_device_context gradient_accumulator)
  cc_test(
    capi_test
    SRCS custom/capi_test.cc
    DEPS phi_capi)
endif()

set(COMM_UTILS_DEPS process_group)
if(WITH_NCCL OR WITH_RCCL)
  set(COMM_UTILS_DEPS ${PROCESS_GROUP_UTILS_DEPS} process_group_nccl)
endif()
if(WITH_CUSTOM_DEVICE)
  set(COMM_UTILS_DEPS ${PROCESS_GROUP_UTILS_DEPS} process_group_custom)
endif()
cc_library(
  processgroup_comm_utils
  SRCS processgroup_comm_utils.cc
  DEPS ${COMM_UTILS_DEPS})
