add_subdirectory(dynload)
add_subdirectory(gpu)

set(BACKENDS_SRCS all_context.cc cpu/cpu_context.cc cpu/cpu_info.cc)
set(BACKENDS_DEPS enforce place flags eigen3 phi_device_context)
if(WITH_XBYAK)
  list(APPEND BACKENDS_DEPS xbyak)
endif()

if(WITH_GPU OR WITH_ROCM)
  list(APPEND BACKENDS_SRCS gpu/gpu_context.cc gpu/gpu_info.cc
       gpu/gpu_resources.cc)
  if(WITH_GPU)
    list(APPEND BACKENDS_SRCS gpu/cuda/cuda_info.cc gpu/cuda/cuda_graph.cc)
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
