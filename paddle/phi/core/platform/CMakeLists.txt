set(DEVICE_SRCS "")

list(APPEND DEVICE_SRCS collective_helper.cc profiler.cc)

if(WITH_GPU OR WITH_ROCM)
  list(APPEND DEVICE_SRCS device/gpu/gpu_info.cc profiler.cu
       device/gpu/gpu_resource_pool.cc)
  list(APPEND DEVICE_SRCS stream_callback_manager.cc)
endif()

if(WITH_GPU)
  list(APPEND DEVICE_SRCS device/gpu/cuda/cuda_profiler.cc)
endif()

if(WITH_XPU)
  list(APPEND DEVICE_SRCS device/xpu/xpu_info.cc
       device/xpu/xpu_resource_pool.cc device/xpu/xpu_op_list.cc)
endif()

if(WITH_CUSTOM_DEVICE)
  list(APPEND DEVICE_SRCS device/custom/custom_device_resource_pool.cc)
endif()

# device event library
list(APPEND DEVICE_SRCS device_event_base.cc)
if(WITH_GPU OR WITH_ROCM)
  list(APPEND DEVICE_SRCS device_event_gpu.cc)
endif()
if(WITH_CUSTOM_DEVICE)
  list(APPEND DEVICE_SRCS device_event_custom_device.cc)
endif()

list(APPEND DEVICE_SRCS cuda_graph_with_memory_pool.cc)
list(APPEND DEVICE_SRCS device_context.cc gen_comm_id_helper.cc)
list(APPEND DEVICE_SRCS profiler/utils.cc profiler/cpu_utilization.cc)

list(APPEND DEVICE_SRCS cpu_helper.cc denormal.cc monitor.cc timer.cc)

collect_srcs(core_srcs SRCS ${DEVICE_SRCS})
