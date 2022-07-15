add_subdirectory(dynload)

set(BACKENDS_SRCS all_context.cc cpu/cpu_context.cc)
set(BACKENDS_DEPS enforce place flags)

if(WITH_GPU OR WITH_ROCM)
  list(APPEND BACKENDS_SRCS gpu/gpu_context.cc gpu/gpu_info.cc
       gpu/gpu_resources.cc)
  if(WITH_GPU)
    list(APPEND BACKENDS_SRCS gpu/cuda/cuda_info.cc)
  endif()
  if(WITH_ROCM)
    list(APPEND BACKENDS_SRCS gpu/rocm/rocm_info.cc)
  endif()
  list(APPEND BACKENDS_DEPS phi_dynload_cuda)
endif()

if(WITH_XPU)
  list(APPEND BACKENDS_SRCS xpu/xpu_context.cc xpu/xpu_info.cc)
endif()

if(WITH_MKLDNN)
  list(APPEND BACKENDS_SRCS onednn/onednn_context.cc)
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
