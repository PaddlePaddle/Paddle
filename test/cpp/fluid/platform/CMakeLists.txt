add_subdirectory(device)
add_subdirectory(profiler)

cc_test(
  enforce_test
  SRCS enforce_test.cc
  DEPS phi common)
cc_test(
  place_test
  SRCS place_test.cc
  DEPS glog phi common)
cc_test(
  errors_test
  SRCS errors_test.cc
  DEPS phi common)
cc_test(
  cpu_info_test
  SRCS cpu_info_test.cc
  DEPS phi common)
cc_test(
  os_info_test
  SRCS os_info_test.cc
  DEPS phi common)
cc_test(
  cpu_helper_test
  SRCS cpu_helper_test.cc
  DEPS phi common)
cc_test(
  init_test
  SRCS init_test.cc
  DEPS device_context)

if(WITH_GPU)
  nv_test(
    device_event_test
    SRCS device_event_test.cc
    DEPS phi common)
  nv_test(
    device_context_test
    SRCS device_context_test.cu
    DEPS device_context phi common)
  nv_test(
    device_context_test_cuda_graph
    SRCS device_context_test_cuda_graph.cu
    DEPS device_context phi common)
endif()

if(WITH_ROCM)
  hip_test(
    device_event_test
    SRCS device_event_test.cc
    DEPS phi common)
  hip_test(
    device_context_test
    SRCS device_context_test.cu
    DEPS device_context phi common)
endif()

cc_test(
  timer_test
  SRCS timer_test.cc
  DEPS phi common)

cc_test(
  lodtensor_printer_test
  SRCS lodtensor_printer_test.cc
  DEPS densetensor_printer)

cc_test(
  profiler_test
  SRCS profiler_test.cc
  DEPS phi common)
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
    DEPS phi common)
endif()

if(WITH_ROCM)
  hip_test(
    float16_gpu_test
    SRCS float16_test.cu
    DEPS lod_tensor)
  hip_test(
    test_limit_gpu_memory
    SRCS test_limit_gpu_memory.cu
    DEPS phi common)
endif()

if(NOT APPLE AND NOT WIN32)
  if(WITH_GPU OR WITH_ROCM)
    cc_test(
      device_code_test
      SRCS device_code_test.cc
      DEPS phi common lod_tensor)
  endif()
endif()

cc_test(
  init_phi_test
  SRCS init_phi_test.cc
  DEPS phi
       common
       init_phi
       op_dialect
       op_dialect_vjp
       static_prim_api
       primitive_backend_static_experimental)
