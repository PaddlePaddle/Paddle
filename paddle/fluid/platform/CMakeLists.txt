proto_library(profiler_proto SRCS profiler.proto DEPS framework_proto simple_threadpool)
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

if(WITH_GPU)
  nv_library(enforce SRCS enforce.cc)
else()
  cc_library(enforce SRCS enforce.cc)
endif()
cc_test(enforce_test SRCS enforce_test.cc DEPS stringpiece enforce)

set(CPU_INFO_DEPS gflags glog enforce)
IF(WITH_XBYAK)
    list(APPEND CPU_INFO_DEPS xbyak)
ENDIF()
cc_library(cpu_info SRCS cpu_info.cc DEPS ${CPU_INFO_DEPS})
cc_test(cpu_info_test SRCS cpu_info_test.cc DEPS cpu_info)

nv_library(gpu_info SRCS gpu_info.cc DEPS gflags glog enforce)

cc_library(place SRCS place.cc DEPS enforce boost)
cc_test(place_test SRCS place_test.cc DEPS place glog gflags)

add_subdirectory(dynload)

cc_library(cpu_helper SRCS cpu_helper.cc DEPS cblas enforce)
cc_test(cpu_helper_test SRCS cpu_helper_test.cc DEPS cpu_helper)

IF(WITH_GPU)
    set(GPU_CTX_DEPS dynload_cuda dynamic_loader)
ELSE()
    set(GPU_CTX_DEPS)
ENDIF()

IF(WITH_MKLDNN)
    set(MKLDNN_CTX_DEPS mkldnn)
ELSE()
    set(MKLDNN_CTX_DEPS)
ENDIF()

cc_library(temp_allocator SRCS temporary_allocator.cc DEPS  allocator_facade)

nv_library(stream_callback_manager SRCS stream_callback_manager.cc DEPS simple_threadpool enforce) 
IF(WITH_GPU)
  set(STREAM_CALLBACK_DEPS stream_callback_manager)
ELSE()
  set(STREAM_CALLBACK_DEPS)
ENDIF()

# memcpy depends on device_context, here add deps individually for
# avoiding cycle dependencies
cc_library(device_context SRCS device_context.cc init.cc DEPS simple_threadpool malloc ${STREAM_CALLBACK_DEPS}
    place eigen3 stringpiece cpu_helper cpu_info framework_proto ${GPU_CTX_DEPS} ${MKLDNN_CTX_DEPS}  temp_allocator)

if(WIN32)
    if(WITH_GPU AND NOT WITH_DSO)
        get_property(cuda_modules GLOBAL PROPERTY CUDA_MODULES)
        target_link_libraries(device_context ${cuda_modules})
    endif(WITH_GPU AND NOT WITH_DSO)
endif(WIN32)

nv_test(device_context_test SRCS device_context_test.cu DEPS device_context gpu_info)

cc_test(init_test SRCS init_test.cc DEPS device_context)

nv_test(cudnn_helper_test SRCS cudnn_helper_test.cc DEPS dynload_cuda)
nv_test(cudnn_desc_test SRCS cudnn_desc_test.cc DEPS dynload_cuda)
nv_test(transform_test SRCS transform_test.cu DEPS memory place device_context)

cc_library(timer SRCS timer.cc)
cc_test(timer_test SRCS timer_test.cc DEPS timer)

cc_library(lodtensor_printer SRCS lodtensor_printer.cc)
cc_test(lodtensor_printer_test SRCS lodtensor_printer_test.cc DEPS lodtensor_printer)

cc_library(device_tracer SRCS device_tracer.cc DEPS boost profiler_proto framework_proto ${GPU_CTX_DEPS})
if(WITH_GPU)
    nv_library(profiler SRCS profiler.cc profiler.cu DEPS device_tracer gpu_info enforce)
else()
    cc_library(profiler SRCS profiler.cc DEPS device_tracer enforce)
endif()
cc_test(profiler_test SRCS profiler_test.cc DEPS profiler)

nv_test(float16_gpu_test SRCS float16_test.cu DEPS lod_tensor)
cc_test(float16_test SRCS float16_test.cc DEPS lod_tensor)

IF(WITH_GPU)
  nv_test(cuda_helper_test SRCS cuda_helper_test.cu)
ENDIF()
nv_library(cuda_device_guard SRCS cuda_device_guard.cc DEPS gpu_info)

if(WITH_GPU)
    nv_test(temporal_allocator_test SRCS temporary_allocator_test.cc DEPS temp_allocator tensor operator)
else()
    cc_test(temporal_allocator_test SRCS temporary_allocator_test.cc DEPS temp_allocator tensor operator)
endif()
