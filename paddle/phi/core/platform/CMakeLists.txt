collect_srcs(core_srcs SRCS collective_helper.cc profiler.cc
             device/gpu/gpu_resource_pool.cc)

if(WITH_GPU OR WITH_ROCM)
  collect_srcs(core_srcs SRCS device/gpu/gpu_info.cc profiler.cu)
endif()

if(WITH_XPU)
  collect_srcs(core_srcs SRCS device/xpu/xpu_info.cc)
endif()
