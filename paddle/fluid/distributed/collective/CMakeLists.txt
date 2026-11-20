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

# gin_overlap: the sharding-stage1 gradient communication built on NCCL GIN
# one-sided puts. Gated on the two things it cannot do without:
#   * a GPU arch it is supported on. It only pays off next to DeepEP, so it
#     targets the same GPUs: 9.0 / 10.0, plus 10.3 which runs the sm_100
#     cubin. The flag is computed from NVCC_ARCH_BIN the way
#     paddle/phi/kernels/CMakeLists.txt computes has_arch_ge80.
#   * NCCL 2.30.7 or newer, with nccl_device.h next to nccl.h: the device
#     (LSA/GIN) API the kernels are written against. Below that the feature is
#     refused rather than silently degraded, same floor as the zero-SM paths
#     (see phi/backends/dynload/nccl.h).
#   * CUDA 12.8 or newer: the intra-node copies go through the batch memcpy
#     API (cudaMemcpyBatchAsync), which appeared in that toolkit.
# The result is cached so that paddle/fluid/pybind/CMakeLists.txt links the
# library and defines PADDLE_WITH_GIN_OVERLAP on exactly the builds that have
# the target, without restating the condition.
set(has_arch_ge90 FALSE)
foreach(arch ${NVCC_ARCH_BIN})
  if(${arch} GREATER_EQUAL 90)
    set(has_arch_ge90 TRUE)
    break()
  endif()
endforeach()

set(gin_overlap_supported FALSE)
if(WITH_DISTRIBUTE
   AND WITH_GPU
   AND WITH_NCCL
   AND has_arch_ge90
   AND (CUDA_VERSION VERSION_GREATER_EQUAL 12.8)
   AND DEFINED NCCL_VERSION)
  if(NCCL_VERSION GREATER_EQUAL 23007 AND EXISTS
                                          "${NCCL_INCLUDE_DIR}/nccl_device.h")
    set(gin_overlap_supported TRUE)
  endif()
endif()
set(WITH_GIN_OVERLAP
    ${gin_overlap_supported}
    CACHE INTERNAL "gin_overlap gradient communication is built")
if(WITH_GIN_OVERLAP)
  message(STATUS "Enable gin_overlap sharding-stage1 gradient communication.")
  add_subdirectory(gin_overlap)
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
