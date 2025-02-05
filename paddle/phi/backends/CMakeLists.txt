add_subdirectory(dynload)
add_subdirectory(gpu)

set(BACKENDS_SRCS all_context.cc cpu/cpu_context.cc cpu/cpu_info.cc)

if(NOT APPLE AND NOT WIN32)
  list(APPEND BACKENDS_SRCS device_code.cc)
endif()

if(WITH_GPU OR WITH_ROCM)
  list(APPEND BACKENDS_SRCS gpu/gpu_context.cc gpu/gpu_info.cc
       gpu/gpu_resources.cc)
  if(WITH_GPU)
    list(APPEND BACKENDS_SRCS gpu/cuda/cuda_info.cc gpu/cuda/cuda_graph.cc)
  endif()
  if(WITH_ROCM)
    list(APPEND BACKENDS_SRCS gpu/rocm/rocm_info.cc gpu/rocm/hip_graph.cc)
  endif()
endif()

if(WITH_XPU)
  list(APPEND BACKENDS_SRCS xpu/xpu_context.cc xpu/xpu_info.cc)
  list(
    APPEND
    BACKENDS_SRCS
    xpu/xpu_op_list.cc
    xpu/xpu1_op_list.cc
    xpu/xpu2_op_list.cc
    xpu/xpu3_op_list.cc
    xpu/xpu_l3_strategy.cc)
endif()

if(WITH_ONEDNN)
  list(APPEND BACKENDS_SRCS onednn/onednn_context.cc)
  list(APPEND BACKENDS_SRCS onednn/axpy_handler.cc)
  list(APPEND BACKENDS_SRCS onednn/matmul_utils.cc)
endif()

list(
  APPEND
  BACKENDS_SRCS
  callback_manager.cc
  device_guard.cc
  stream.cc
  event.cc
  device_manager.cc
  context_pool.cc)

if(WITH_GPU
   OR WITH_ROCM
   OR WITH_CUSTOM_DEVICE)
  list(APPEND BACKENDS_SRCS device_base.cc)
endif()

if(WITH_CUSTOM_DEVICE)
  list(APPEND BACKENDS_SRCS custom/custom_context.cc custom/custom_device.cc
       custom/custom_device_op_list.cc)
endif()

collect_srcs(backends_srcs SRCS ${BACKENDS_SRCS})
