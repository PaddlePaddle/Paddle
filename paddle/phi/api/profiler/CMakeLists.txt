proto_library(phi_profiler_proto SRCS profiler.proto)

if(WITH_PYTHON AND EXISTS ${PADDLE_BINARY_DIR})
  set(FLUID_PATH ${PADDLE_BINARY_DIR}/python/paddle/fluid)
  py_proto_compile(profiler_py_proto SRCS profiler.proto)
  file(TOUCH ${CMAKE_CURRENT_BINARY_DIR}/__init__.py)
  if(NOT WIN32)
    add_custom_command(
      TARGET profiler_py_proto
      POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E make_directory ${FLUID_PATH}/proto/profiler
      COMMAND cp *.py ${FLUID_PATH}/proto/profiler
      COMMENT
        "Copy generated python proto into directory paddle/fluid/proto/profiler."
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  else()
    string(REPLACE "/" "\\" proto_dstpath "${FLUID_PATH}/proto/profiler/")
    add_custom_command(
      TARGET profiler_py_proto
      POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E make_directory ${FLUID_PATH}/proto/profiler
      COMMAND copy /Y *.py ${proto_dstpath}
      COMMENT
        "Copy generated python proto into directory paddle/fluid/proto/profiler."
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  endif()
endif()

if(WITH_GPU OR WITH_ROCM)
  set(GPU_CTX_DEPS dynload_cuda dynamic_loader)
endif()

cc_library(
  phi_device_tracer
  SRCS device_tracer.cc
  DEPS phi_profiler_proto ${GPU_CTX_DEPS})

cc_library(
  phi_profiler
  SRCS profiler.cc
  DEPS phi_os_info phi_device_tracer phi_enforce)
