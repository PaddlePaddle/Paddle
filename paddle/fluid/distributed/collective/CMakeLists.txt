string(FIND "${COMPILED_CUDA_ARCHS}" "90" ARCH_BIN_CONTAINS_90)

if(WITH_DISTRIBUTE
   AND WITH_GPU
   AND (ARCH_BIN_CONTAINS_90 GREATER -1))
  message(STATUS "Enable DeepEP alltoall communication.")
  if(WITH_NVSHMEM)
    set(CMAKE_CUDA_FLAGS
        "${CMAKE_CUDA_FLAGS} -rdc=true --ptxas-options=--register-usage-level=10,--warn-on-local-memory-usage"
    )
    string(REGEX REPLACE "-gencode arch=compute_[0-8][0-9],code=sm_[0-8][0-9]"
                         "" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")

    set(DEEPEP_KERNEL_SRCS
        deep_ep/kernels/intranode.cu
        deep_ep/kernels/runtime.cu
        deep_ep/kernels/internode.cu
        deep_ep/kernels/internode_ll.cu
        deep_ep/kernels/internode_ll_two_stage.cu
        flash_ep/kernels/runtime.cu
        flash_ep/kernels/internode.cu)
    cc_library(
      deepep_kernels
      SRCS ${DEEPEP_KERNEL_SRCS}
      DEPS nvshmem cudadevrt)

    set_target_properties(deepep_kernels PROPERTIES CUDA_SEPARABLE_COMPILATION
                                                    ON)
    set_target_properties(deepep_kernels PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS
                                                    ON)
  else()
    set(DEEPEP_KERNEL_SRCS
        deep_ep/kernels/intranode.cu deep_ep/kernels/runtime.cu
        flash_ep/kernels/runtime.cu)
    cc_library(deepep_kernels SRCS ${DEEPEP_KERNEL_SRCS})
  endif()

  cc_library(
    deep_ep
    SRCS deep_ep/deep_ep.cpp
         deep_ep/src/event_pool.cc
         deep_ep/src/event.cc
         deep_ep/src/CUDAStream.cc
         flash_ep/flash_ep.cpp
         flash_ep/src/event_pool.cc
         flash_ep/src/event.cc
         flash_ep/src/CUDAStream.cc
    DEPS phi common deepep_kernels)

  set_target_properties(deep_ep PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
  target_compile_options(deep_ep PRIVATE -Wno-reorder -Wno-unused-variable)

  set(DEEPEP_LIB deep_ep)
endif()

cc_library(
  process_group
  SRCS process_group.cc
  DEPS phi common xxhash ${DEEPEP_LIB})

cc_library(
  eager_reducer
  SRCS reducer.cc
  DEPS eager_api process_group phi common string_helper)

if(WITH_DISTRIBUTE)
  cc_library(
    process_group_gloo
    SRCS process_group_gloo.cc gloo_send_recv.cc
    DEPS phi common eager_api gloo_wrapper)
endif()

if(WITH_NCCL OR WITH_RCCL)
  cc_library(
    process_group_nccl
    SRCS process_group_nccl.cc common.cc
    DEPS process_group phi ${DEVICE_EVENT_LIBS})

  cc_library(
    async_load
    SRCS async_load.cc
    DEPS phi ${DEVICE_EVENT_LIBS})

endif()

if(WITH_FLAGCX AND NOT WITH_XPU)
  cc_library(
    process_group_flagcx
    SRCS process_group_flagcx.cc common.cc
    DEPS process_group phi)
endif()

if(WITH_XPU_BKCL)
  cc_library(
    process_group_bkcl
    SRCS process_group_bkcl.cc bkcl_tools.cc common.cc
         process_group_kernel_utils.cc
    DEPS process_group phi)

  cc_library(
    xpu_async_load
    SRCS xpu_async_load.cc
    DEPS process_group phi ${DEVICE_EVENT_LIBS})
endif()

if(WITH_MPI)
  cc_library(
    process_group_mpi
    SRCS process_group_mpi.cc mpi_tools.cc common.cc
    DEPS phi)
endif()

if(WITH_CUSTOM_DEVICE)
  cc_library(
    process_group_custom
    SRCS process_group_custom.cc custom_ccl_tools.cc common.cc
    DEPS process_group phi)
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
