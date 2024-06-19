if(WITH_GPU)
  set(HETERPS_DEPS device_context)
  if(${CMAKE_CUDA_COMPILER_VERSION} LESS 11.0)
    set(HETERPS_DEPS ${HETERPS_DEPS} cub)
  endif()
  if(WITH_PSCORE)
    get_property(RPC_DEPS GLOBAL PROPERTY RPC_DEPS)
    set(HETERPS_DEPS ${HETERPS_DEPS} ${RPC_DEPS})
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
  if(WITH_HETERPS)
    nv_library(
      graph_gpu_ps
      SRCS graph_gpu_ps_table_inl.cu
      DEPS heter_comm table hashtable_kernel)
    nv_library(
      graph_sampler
      SRCS graph_sampler_inl.h
      DEPS graph_gpu_ps)
    nv_library(
      graph_gpu_wrapper
      SRCS graph_gpu_wrapper.cu
      DEPS heter_comm
           table
           heter_comm_kernel
           hashtable_kernel
           heter_ps
           ${HETERPS_DEPS}
           graph_gpu_ps
           fleet_wrapper
           gloo_wrapper)
    #nv_test(
    #  test_cpu_query
    #  SRCS test_cpu_query.cu
    #  DEPS heter_comm
    #       table
    #       heter_comm_kernel
    #       hashtable_kernel
    #       heter_ps
    #       ${HETERPS_DEPS}
    #       graph_gpu_ps
    #       graph_gpu_wrapper)
    #ADD_EXECUTABLE(test_sample_rate test_sample_rate.cu)
    #target_link_libraries(test_sample_rate heter_comm table heter_comm_kernel hashtable_kernel heter_ps ${HETERPS_DEPS})
    #nv_test(test_sample_rate SRCS test_sample_rate.cu DEPS heter_comm table heter_comm_kernel hashtable_kernel heter_ps ${HETERPS_DEPS})
    #ADD_EXECUTABLE(test_cpu_query test_cpu_query.cu)
    #target_link_libraries(test_cpu_query graph_gpu_ps)
  endif()
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
