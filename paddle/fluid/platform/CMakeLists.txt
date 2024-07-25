cc_library(
  denormal
  SRCS denormal.cc
  DEPS)

cc_test(
  errors_test
  SRCS errors_test.cc
  DEPS phi common enforce)

set(enforce_deps phi)
if(WITH_GPU)
  set(enforce_deps ${enforce_deps} external_error_proto)
endif()

cc_library(
  enforce INTERFACE
  SRCS enforce.cc
  DEPS ${enforce_deps})
cc_library(monitor SRCS monitor.cc)
cc_test(
  enforce_test
  SRCS enforce_test.cc
  DEPS enforce common)

cc_test(
  cpu_info_test
  SRCS cpu_info_test.cc
  DEPS phi common)
cc_test(
  os_info_test
  SRCS os_info_test.cc
  DEPS phi common)

cc_test(
  place_test
  SRCS place_test.cc
  DEPS glog phi common)

if(WITH_ONEDNN)
  set(ONEDNN_CTX_DEPS onednn)
else()
  set(ONEDNN_CTX_DEPS)
endif()

add_subdirectory(device)

cc_library(
  cpu_helper
  SRCS cpu_helper.cc
  DEPS cblas enforce common)
cc_test(
  cpu_helper_test
  SRCS cpu_helper_test.cc
  DEPS cpu_helper)

set(dgc_deps "")
if(WITH_DGC)
  set(dgc_deps dgc)
endif()

if(WITH_GPU OR WITH_ROCM)
  set(GPU_CTX_DEPS phi common)
endif()

if(WITH_XPU)
  set(XPU_CTX_DEPS phi common)
endif()

if(WITH_IPU)
  set(IPU_CTX_DEPS ipu_info)
else()
  set(IPU_CTX_DEPS)
endif()

if(WITH_GPU)
  nv_library(
    stream_callback_manager
    SRCS stream_callback_manager.cc
    DEPS simple_threadpool enforce common)
endif()
if(WITH_ROCM)
  hip_library(
    stream_callback_manager
    SRCS stream_callback_manager.cc
    DEPS simple_threadpool enforce common)
endif()

if(WITH_GPU OR WITH_ROCM)
  set(STREAM_CALLBACK_DEPS stream_callback_manager)
else()
  set(STREAM_CALLBACK_DEPS)
endif()

if(WITH_GLOO)
  cc_library(
    gloo_context
    SRCS gloo_context.cc
    DEPS framework_proto gloo_wrapper enforce common)
endif()

# separate init from device_context to avoid cycle dependencies
cc_library(
  init
  SRCS init.cc
  DEPS device_context phi common fluid_memory)

#fluid_memory depends on device_context, here add deps individually for
# avoiding cycle dependencies

cc_library(
  device_context
  SRCS device_context.cc
  DEPS simple_threadpool
       allocator
       xxhash
       ${STREAM_CALLBACK_DEPS}
       eigen3
       cpu_helper
       framework_proto
       ${IPU_CTX_DEPS}
       ${GPU_CTX_DEPS}
       ${ONEDNN_CTX_DEPS}
       ${dgc_deps}
       dlpack
       phi
       common
       ${XPU_CTX_DEPS})

cc_library(
  collective_helper
  SRCS collective_helper.cc gen_comm_id_helper.cc
  DEPS framework_proto device_context enforce common)

if(WITH_GPU OR WITH_ROCM)
  target_link_libraries(device_context gpu_resource_pool)
endif()

if(WITH_XPU)
  target_link_libraries(device_context xpu_resource_pool)
endif()

if(WITH_CUSTOM_DEVICE)
  target_link_libraries(device_context custom_device_resource_pool)
endif()

cc_test(
  init_test
  SRCS init_test.cc
  DEPS device_context)

# Manage all device event library
set(DEVICE_EVENT_LIBS)
cc_library(
  device_event_base
  SRCS device_event_base.cc
  DEPS enforce device_context op_registry phi common)
set(DEVICE_EVENT_LIBS
    device_event_base
    CACHE INTERNAL "device event libs")

if(WITH_GPU)
  nv_library(
    device_event_gpu
    SRCS device_event_gpu.cc
    DEPS device_event_base)
  set(DEVICE_EVENT_LIBS
      device_event_gpu
      CACHE INTERNAL "device event libs")
  if(WITH_CUSTOM_DEVICE)
    nv_test(
      device_event_test
      SRCS device_event_test.cc
      DEPS device_event_gpu device_event_custom_device)
  else()
    nv_test(
      device_event_test
      SRCS device_event_test.cc
      DEPS device_event_gpu)
  endif()
  if(WITH_CUSTOM_DEVICE)
    nv_library(
      cuda_graph_with_memory_pool
      SRCS cuda_graph_with_memory_pool.cc
      DEPS ${DEVICE_EVENT_LIBS} device_event_custom_device device_context
           allocator phi common)
  else()
    nv_library(
      cuda_graph_with_memory_pool
      SRCS cuda_graph_with_memory_pool.cc
      DEPS ${DEVICE_EVENT_LIBS} device_context allocator phi common)
  endif()
  nv_test(
    device_context_test
    SRCS device_context_test.cu
    DEPS device_context gpu_info)
  nv_test(
    device_context_test_cuda_graph
    SRCS device_context_test_cuda_graph.cu
    DEPS device_context gpu_info cuda_graph_with_memory_pool)
else()
  cc_library(
    cuda_graph_with_memory_pool
    SRCS cuda_graph_with_memory_pool.cc
    DEPS device_context allocator)
endif()

if(WITH_ROCM)
  hip_library(
    device_event_gpu
    SRCS device_event_gpu.cc
    DEPS device_event_base)
  set(DEVICE_EVENT_LIBS
      device_event_gpu
      CACHE INTERNAL "device event libs")
  if(WITH_CUSTOM_DEVICE)
    hip_test(
      device_event_test
      SRCS device_event_test.cc
      DEPS device_event_gpu device_event_custom_device)
  else()
    hip_test(
      device_event_test
      SRCS device_event_test.cc
      DEPS device_event_gpu)
  endif()
  hip_test(
    device_context_test
    SRCS device_context_test.cu
    DEPS device_context gpu_info)
endif()

cc_library(timer SRCS timer.cc)
cc_test(
  timer_test
  SRCS timer_test.cc
  DEPS timer)

cc_library(
  lodtensor_printer
  SRCS lodtensor_printer.cc
  DEPS phi
       common
       tensor
       scope
       lod_tensor
       variable_helper
       framework_proto)
cc_test(
  lodtensor_printer_test
  SRCS lodtensor_printer_test.cc
  DEPS lodtensor_printer)

add_subdirectory(profiler)

if(WITH_GPU)
  nv_library(
    profiler
    SRCS profiler.cc profiler.cu
    DEPS phi common gpu_info enforce op_proto_maker shape_inference)
elseif(WITH_ROCM)
  hip_library(
    profiler
    SRCS profiler.cc profiler.cu
    DEPS phi common gpu_info enforce op_proto_maker shape_inference)
elseif(WITH_XPU)
  cc_library(
    profiler
    SRCS profiler.cc
    DEPS phi common enforce op_proto_maker shape_inference)
else()
  cc_library(
    profiler
    SRCS profiler.cc
    DEPS phi common enforce op_proto_maker shape_inference)
endif()

cc_test(
  profiler_test
  SRCS profiler_test.cc
  DEPS profiler)
cc_test(
  float16_test
  SRCS float16_test.cc
  DEPS lod_tensor)
cc_test(
  bfloat16_test
  SRCS bfloat16_test.cc
  DEPS lod_tensor)
cc_test(
  complex_test
  SRCS complex_test.cc
  DEPS lod_tensor)

if(WITH_GPU)
  nv_test(
    float16_gpu_test
    SRCS float16_test.cu
    DEPS lod_tensor)
  nv_test(
    bfloat16_gpu_test
    SRCS bfloat16_test.cu
    DEPS lod_tensor)
  nv_test(
    complex_gpu_test
    SRCS complex_test.cu
    DEPS lod_tensor)
  nv_test(
    test_limit_gpu_memory
    SRCS test_limit_gpu_memory.cu
    DEPS gpu_info phi common)
endif()

if(WITH_ROCM)
  hip_test(
    float16_gpu_test
    SRCS float16_test.cu
    DEPS lod_tensor)
  hip_test(
    test_limit_gpu_memory
    SRCS test_limit_gpu_memory.cu
    DEPS gpu_info phi common)
endif()

if(NOT APPLE AND NOT WIN32)
  if(WITH_GPU OR WITH_ROCM)
    cc_test(
      device_code_test
      SRCS device_code_test.cc
      DEPS phi common lod_tensor)
  endif()
endif()

if(WITH_CUSTOM_DEVICE)
  cc_library(
    device_event_custom_device
    SRCS device_event_custom_device.cc
    DEPS device_event_base custom_device_resource_pool)
  set(DEVICE_EVENT_LIBS
      ${DEVICE_EVENT_LIBS} device_event_custom_device
      CACHE INTERNAL "device event libs")
endif()

cc_library(
  init_phi
  SRCS init_phi.cc
  DEPS init)

cc_test(
  init_phi_test
  SRCS init_phi_test.cc
  DEPS phi common init_phi)

if(TENSORRT_FOUND)
  add_subdirectory(tensorrt)
endif()
