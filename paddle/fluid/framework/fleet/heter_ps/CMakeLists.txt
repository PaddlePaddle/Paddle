if(WITH_GPU)
  set(HETERPS_DEPS device_context)
  if(${CMAKE_CUDA_COMPILER_VERSION} LESS 11.0)
    set(HETERPS_DEPS ${HETERPS_DEPS} cub)
  endif()
  nv_library(
    heter_comm_kernel
    SRCS heter_comm_kernel.cu feature_value.h feature_value.cu
    DEPS ${HETERPS_DEPS})
  nv_library(
    hashtable_kernel
    SRCS hashtable_kernel.cu feature_value.h feature_value.cu
    DEPS ${HETERPS_DEPS})
  nv_library(
    heter_comm
    SRCS heter_comm.h feature_value.h feature_value.cu heter_resource.cc
         heter_resource.h mem_pool.h
    DEPS ${HETERPS_DEPS} heter_comm_kernel hashtable_kernel)
  nv_test(
    test_heter_comm
    SRCS feature_value.h
    DEPS heter_comm)
  nv_library(
    heter_ps
    SRCS heter_ps.cu
    DEPS heter_comm)
endif()
if(WITH_XPU_KP)
  set(HETERPS_DEPS device_context)
  xpu_library(heter_comm_kernel SRCS heter_comm_kernel.h heter_comm_kernel.kps
                                     feature_value.h)
  xpu_library(hashtable_kernel SRCS hashtable.h hashtable_kernel.kps)
  cc_library(
    heter_comm
    SRCS heter_comm.h heter_resource.cc
    DEPS ${HETERPS_DEPS} heter_comm_kernel hashtable_kernel)
  cc_library(
    heter_ps
    SRCS heter_ps.cc
    DEPS heter_comm)
endif()
if(WITH_ROCM)
  hip_library(
    heter_comm
    SRCS heter_comm.h feature_value.h heter_resource.cc heter_resource.h
         hashtable.h
    DEPS cub device_context)
  hip_test(
    test_heter_comm
    SRCS feature_value.h
    DEPS heter_comm)
  hip_library(
    heter_ps
    SRCS heter_ps.cu
    DEPS heter_comm)
endif()
