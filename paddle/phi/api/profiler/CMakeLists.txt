proto_library(phi_profiler_proto SRCS profiler.proto)

if(WITH_PYTHON AND EXISTS ${PADDLE_BINARY_DIR})
  set(FLUID_PATH ${PADDLE_BINARY_DIR}/python/paddle/base)
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

collect_srcs(api_srcs SRCS device_tracer.cc profiler.cc)
