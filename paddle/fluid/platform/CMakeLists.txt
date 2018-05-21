proto_library(profiler_proto SRCS profiler.proto DEPS framework_proto)
py_proto_compile(profiler_py_proto SRCS profiler.proto)

add_custom_target(profiler_py_proto_init ALL COMMAND ${CMAKE_COMMAND} -E touch __init__.py)

add_dependencies(profiler_py_proto profiler_py_proto_init)

add_custom_command(TARGET profiler_py_proto POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${PADDLE_BINARY_DIR}/python/paddle/fluid/proto/profiler
        COMMAND cp *.py ${PADDLE_BINARY_DIR}/python/paddle/fluid/proto/profiler
        COMMENT "Copy generated python proto into directory paddle/fluid/proto/profiler."
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

if(WITH_GPU)
  nv_library(enforce SRCS enforce.cc)
else()
  cc_library(enforce SRCS enforce.cc)
endif()
cc_test(enforce_test SRCS enforce_test.cc DEPS stringpiece enforce)

cc_library(cpu_info SRCS cpu_info.cc DEPS gflags glog enforce)
cc_test(cpu_info_test SRCS cpu_info_test.cc DEPS cpu_info)

nv_library(gpu_info SRCS gpu_info.cc DEPS gflags glog enforce)

cc_library(place SRCS place.cc DEPS enforce boost)
cc_test(place_test SRCS place_test.cc DEPS place glog gflags)

add_subdirectory(dynload)

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

# memcpy depends on device_context, here add deps individually for
# avoiding cycle dependencies
cc_library(device_context SRCS device_context.cc DEPS malloc
    place eigen3 ${GPU_CTX_DEPS} ${MKLDNN_CTX_DEPS})
nv_test(device_context_test SRCS device_context_test.cu DEPS device_context gpu_info)

nv_test(cudnn_helper_test SRCS cudnn_helper_test.cc DEPS dynload_cuda)
nv_test(transform_test SRCS transform_test.cu DEPS memory place device_context)

cc_library(device_tracer SRCS device_tracer.cc DEPS boost profiler_proto framework_proto ${GPU_CTX_DEPS})
cc_library(profiler SRCS profiler.cc DEPS device_context device_tracer)
cc_test(profiler_test SRCS profiler_test.cc DEPS profiler)

nv_test(float16_gpu_test SRCS float16_test.cu)
cc_test(float16_test SRCS float16_test.cc)
