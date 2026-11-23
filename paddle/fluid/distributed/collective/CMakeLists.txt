string(REGEX MATCH "9[0-9]|1[0-9][0-9]" ARCH_BIN_CONTAINS_GE_90
             "${COMPILED_CUDA_ARCHS}")

if(WITH_DISTRIBUTE
   AND WITH_GPU
   AND ARCH_BIN_CONTAINS_GE_90)
  message(STATUS "Enable DeepEP alltoall communication.")
  add_subdirectory(deep_ep)
  set(DEEPEP_LIB deep_ep)
endif()

if(WITH_DISTRIBUTE
   AND WITH_XPU
   AND WITH_XPU_XRE5)
  message(STATUS "Enable DeepEP alltoall communication.")
  add_subdirectory(deep_ep_xpu)
  set(DEEPEP_LIB deep_ep_xpu)
endif()

# NCCL one-sided primitives of ProcessGroupNCCL (GIN puts, LSA counters and
# copies). Built only when all of these hold:
#   * WITH_DISTRIBUTE, WITH_GPU and WITH_NCCL.
#   * An arch >= 9.0 is compiled (ARCH_BIN_CONTAINS_GE_90, as for DeepEP).
#   * NCCL 2.30.7 or newer, with nccl_device.h next to nccl.h (the device
#     LSA/GIN API).
#   * CUDA 13.0 or newer: the code calls cudaMemcpyBatchAsync with the
#     CUDA 13 signature (CUDA 12.8 / 12.9 declare a different one).
# Without them the ProcessGroupNCCL methods still exist and raise Unavailable.
set(nccl_one_sided_supported FALSE)
if(WITH_DISTRIBUTE
   AND WITH_GPU
   AND WITH_NCCL
   AND ARCH_BIN_CONTAINS_GE_90
   AND (CUDA_VERSION VERSION_GREATER_EQUAL 13.0)
   AND DEFINED NCCL_VERSION)
  if(NCCL_VERSION GREATER_EQUAL 23007 AND EXISTS
                                          "${NCCL_INCLUDE_DIR}/nccl_device.h")
    set(nccl_one_sided_supported TRUE)
  endif()
endif()
set(NCCL_ONE_SIDED_LIB "")
if(nccl_one_sided_supported)
  message(STATUS "Enable NCCL one-sided primitives of ProcessGroupNCCL.")
  add_subdirectory(gin_overlap/kernels)
  set(NCCL_ONE_SIDED_LIB gin_overlap_comm_kernels)
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
    DEPS process_group phi ${DEVICE_EVENT_LIBS} ${NCCL_ONE_SIDED_LIB})
  if(nccl_one_sided_supported)
    target_compile_definitions(process_group_nccl
                               PRIVATE PADDLE_WITH_NCCL_ONE_SIDED)
  endif()

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
