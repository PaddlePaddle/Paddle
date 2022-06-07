proto_library(profiler_proto SRCS profiler.proto DEPS framework_proto simple_threadpool)
if(WITH_GPU)
  proto_library(external_error_proto SRCS external_error.proto)
endif(WITH_GPU)
if (WITH_PYTHON)
  py_proto_compile(profiler_py_proto SRCS profiler.proto)
  add_custom_target(profiler_py_proto_init ALL COMMAND ${CMAKE_COMMAND} -E touch __init__.py)
  add_dependencies(profiler_py_proto profiler_py_proto_init)

  if (NOT WIN32)
    add_custom_command(TARGET profiler_py_proto POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${PADDLE_BINARY_DIR}/python/paddle/fluid/proto/profiler
        COMMAND cp *.py ${PADDLE_BINARY_DIR}/python/paddle/fluid/proto/profiler
        COMMENT "Copy generated python proto into directory paddle/fluid/proto/profiler."
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  else(NOT WIN32)
    string(REPLACE "/" "\\" proto_dstpath "${PADDLE_BINARY_DIR}/python/paddle/fluid/proto/profiler/")
    add_custom_command(TARGET profiler_py_proto POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${PADDLE_BINARY_DIR}/python/paddle/fluid/proto/profiler
        COMMAND copy /Y *.py ${proto_dstpath}
        COMMENT "Copy generated python proto into directory paddle/fluid/proto/profiler."
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  endif(NOT WIN32)
endif()

cc_library(flags SRCS flags.cc DEPS gflags boost)
cc_library(denormal SRCS denormal.cc DEPS)

cc_test(errors_test SRCS errors_test.cc DEPS errors enforce)

set(enforce_deps flags errors boost flags phi_enforce)
if(WITH_GPU)
  set(enforce_deps ${enforce_deps} external_error_proto)
endif()

cc_library(enforce INTERFACE SRCS enforce.cc DEPS ${enforce_deps})
cc_library(monitor SRCS monitor.cc)
cc_test(enforce_test SRCS enforce_test.cc DEPS stringpiece enforce)

set(CPU_INFO_DEPS gflags glog enforce)
IF(WITH_XBYAK)
    list(APPEND CPU_INFO_DEPS xbyak)
ENDIF()
cc_library(cpu_info SRCS cpu_info.cc DEPS ${CPU_INFO_DEPS})
cc_test(cpu_info_test SRCS cpu_info_test.cc DEPS cpu_info)
cc_library(os_info SRCS os_info.cc DEPS enforce)
cc_test(os_info_test SRCS os_info_test.cc DEPS os_info)

IF(WITH_GPU)
    nv_library(cuda_graph_with_memory_pool SRCS cuda_graph_with_memory_pool.cc DEPS device_context allocator_facade cuda_graph)
ELSE()
    cc_library(cuda_graph_with_memory_pool SRCS cuda_graph_with_memory_pool.cc DEPS device_context allocator_facade)
ENDIF()

cc_library(place SRCS place.cc DEPS enforce boost phi_place)
cc_test(place_test SRCS place_test.cc DEPS place glog gflags)

IF(WITH_MKLDNN)
    set(MKLDNN_CTX_DEPS mkldnn)
ELSE()
    set(MKLDNN_CTX_DEPS)
ENDIF()

add_subdirectory(device)
add_subdirectory(dynload)
add_subdirectory(stream)

cc_library(cpu_helper SRCS cpu_helper.cc DEPS cblas enforce)
cc_test(cpu_helper_test SRCS cpu_helper_test.cc DEPS cpu_helper)

set(dgc_deps "")
IF(WITH_DGC)
    set(dgc_deps dgc)
ENDIF()

IF(WITH_GPU OR WITH_ROCM)
    set(GPU_CTX_DEPS dynload_cuda dynamic_loader cuda_stream)
ENDIF()

IF(WITH_IPU)
    set(IPU_CTX_DEPS ipu_info)
ELSE()
    set(IPU_CTX_DEPS)
ENDIF(WITH_IPU)

IF(WITH_ASCEND_CL)
    set(NPU_CTX_DEPS npu_stream npu_info)
ENDIF()

IF(WITH_MLU)
    set(MLU_CTX_DEPS mlu_device_context)
ENDIF()

IF(WITH_ASCEND_CL OR WITH_MLU)
cc_library(stream_callback_manager SRCS stream_callback_manager.cc DEPS simple_threadpool enforce)
ENDIF()

IF(WITH_GPU)
    nv_library(stream_callback_manager SRCS stream_callback_manager.cc DEPS simple_threadpool enforce)
ENDIF()
IF(WITH_ROCM)
    hip_library(stream_callback_manager SRCS stream_callback_manager.cc DEPS simple_threadpool enforce)
ENDIF()

IF(WITH_GPU OR WITH_ROCM)
  set(STREAM_CALLBACK_DEPS stream_callback_manager)
ELSEIF(WITH_ASCEND_CL)
  set(STREAM_CALLBACK_DEPS stream_callback_manager)
ELSE()
  set(STREAM_CALLBACK_DEPS)
ENDIF()

if(WITH_GLOO)
    cc_library(gloo_context SRCS gloo_context.cc DEPS framework_proto gloo_wrapper enforce)
endif()

cc_library(cudnn_workspace_helper SRCS cudnn_workspace_helper.cc DEPS boost)

# seperate init from device_context to avoid cycle dependencies
cc_library(init SRCS init.cc DEPS device_context custom_kernel context_pool)

# memcpy depends on device_context, here add deps individually for
# avoiding cycle dependencies
cc_library(device_context SRCS device_context.cc DEPS simple_threadpool malloc xxhash ${STREAM_CALLBACK_DEPS}
    place phi_place eigen3 stringpiece cpu_helper cpu_info framework_proto ${IPU_CTX_DEPS} ${GPU_CTX_DEPS} ${NPU_CTX_DEPS} ${MKLDNN_CTX_DEPS}
    ${dgc_deps} dlpack cudnn_workspace_helper ${XPU_CTX_DEPS} ${MLU_CTX_DEPS} eigen3 cpu_context generator)
if(WITH_XPU)
  target_link_libraries(device_context xpu_context)
endif()

cc_library(collective_helper SRCS collective_helper.cc gen_comm_id_helper.cc DEPS framework_proto device_context enforce)
if(WITH_ASCEND_CL)
    target_link_libraries(collective_helper npu_collective_helper)
endif()

if(WITH_CNCL)
    target_link_libraries(collective_helper mlu_collective_helper)
endif()

if(WITH_GPU OR WITH_ROCM)
    target_link_libraries(device_context gpu_info gpu_context phi_gpu_info)
    target_link_libraries(device_context gpu_resource_pool)
endif()
if (WITH_CUSTOM_DEVICE)
    target_link_libraries(device_context custom_context)
endif()
if(WITH_ASCEND_CL)
    target_link_libraries(device_context npu_resource_pool)
endif()

if(WITH_MLU)
    target_link_libraries(device_context mlu_resource_pool)
endif()

if(WITH_CUSTOM_DEVICE)
    target_link_libraries(device_context custom_context)
endif()

cc_test(init_test SRCS init_test.cc DEPS device_context)

# Manage all device event library
set(DEVICE_EVENT_LIBS)
cc_library(device_event_base SRCS device_event_base.cc DEPS place enforce device_context op_registry)
set(DEVICE_EVENT_LIBS  device_event_base CACHE INTERNAL "device event libs")


if(WITH_GPU)
  nv_library(device_event_gpu SRCS device_event_gpu.cc DEPS device_event_base)
  set(DEVICE_EVENT_LIBS  device_event_gpu CACHE INTERNAL "device event libs")
  nv_test(device_event_test SRCS device_event_test.cc DEPS device_event_gpu)

  nv_test(device_context_test SRCS device_context_test.cu DEPS device_context gpu_info)
  nv_test(transform_test SRCS transform_test.cu DEPS memory place device_context)
endif()

if(WITH_ROCM)
  hip_library(device_event_gpu SRCS device_event_gpu.cc DEPS device_event_base)
  set(DEVICE_EVENT_LIBS  device_event_gpu CACHE INTERNAL "device event libs")
  hip_test(device_event_test SRCS device_event_test.cc DEPS device_event_gpu)

  hip_test(device_context_test SRCS device_context_test.cu DEPS device_context gpu_info)
  hip_test(transform_test SRCS transform_test.cu DEPS memory place device_context)
endif()

cc_library(timer SRCS timer.cc)
cc_test(timer_test SRCS timer_test.cc DEPS timer)

cc_library(lodtensor_printer SRCS lodtensor_printer.cc DEPS ddim place tensor scope lod_tensor variable_helper framework_proto)
cc_test(lodtensor_printer_test SRCS lodtensor_printer_test.cc DEPS lodtensor_printer)

add_subdirectory(profiler)

cc_library(device_tracer SRCS device_tracer.cc DEPS boost profiler_proto framework_proto ${GPU_CTX_DEPS})
if(WITH_GPU)
  nv_library(profiler SRCS profiler.cc profiler.cu DEPS os_info device_tracer gpu_info enforce dynload_cuda new_profiler stats)
  nv_library(device_memory_aligment SRCS device_memory_aligment.cc DEPS cpu_info gpu_info place)
elseif(WITH_ROCM)
  hip_library(profiler SRCS profiler.cc profiler.cu DEPS os_info device_tracer gpu_info enforce new_profiler stats)
  hip_library(device_memory_aligment SRCS device_memory_aligment.cc DEPS cpu_info gpu_info place)
else()
  cc_library(profiler SRCS profiler.cc DEPS os_info device_tracer enforce new_profiler stats)
  cc_library(device_memory_aligment SRCS device_memory_aligment.cc DEPS cpu_info place)
endif()

cc_test(profiler_test SRCS profiler_test.cc DEPS profiler)
cc_test(float16_test SRCS float16_test.cc DEPS lod_tensor)
cc_test(bfloat16_test SRCS bfloat16_test.cc DEPS lod_tensor)
cc_test(complex_test SRCS complex_test.cc DEPS lod_tensor)

IF(WITH_GPU)
  nv_test(float16_gpu_test SRCS float16_test.cu DEPS lod_tensor)
  nv_test(bfloat16_gpu_test SRCS bfloat16_test.cu DEPS lod_tensor)
  nv_test(complex_gpu_test SRCS complex_test.cu DEPS lod_tensor)
  nv_test(test_limit_gpu_memory SRCS test_limit_gpu_memory.cu DEPS gpu_info flags)
  nv_library(cuda_device_guard SRCS cuda_device_guard.cc DEPS gpu_info)
ENDIF()

IF(WITH_ROCM)
  hip_test(float16_gpu_test SRCS float16_test.cu DEPS lod_tensor)
  hip_test(test_limit_gpu_memory SRCS test_limit_gpu_memory.cu DEPS gpu_info flags)
  hip_library(cuda_device_guard SRCS cuda_device_guard.cc DEPS gpu_info)
ENDIF()

if(NOT APPLE AND NOT WIN32)
  cc_library(device_code SRCS device_code.cc DEPS device_context)
  if(WITH_GPU OR WITH_ROCM)
    cc_test(device_code_test SRCS device_code_test.cc DEPS device_code lod_tensor)
  endif()
endif()
