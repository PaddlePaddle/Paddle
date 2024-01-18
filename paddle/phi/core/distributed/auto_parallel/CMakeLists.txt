proto_library(auto_parallel_proto SRCS auto_parallel.proto)

if(WITH_PYTHON)
  py_proto_compile(auto_parallel_py_proto SRCS auto_parallel.proto)

  if(NOT WIN32)
    add_custom_command(
      TARGET auto_parallel_py_proto
      POST_BUILD
      COMMAND
        ${CMAKE_COMMAND} -E make_directory
        ${PADDLE_BINARY_DIR}/python/paddle/distributed/auto_parallel/proto/
      COMMAND
        ${CMAKE_COMMAND} -E touch
        ${PADDLE_BINARY_DIR}/python/paddle/distributed/auto_parallel/proto/__init__.py
      COMMAND
        cp auto_parallel_*.py
        ${PADDLE_BINARY_DIR}/python/paddle/distributed/auto_parallel/proto/
      COMMENT
        "Copy generated python proto into directory paddle/distributed/auto_parallel/proto/."
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  else()
    string(
      REPLACE
        "/" "\\" auto_parallel_proto_dstpath
        "${PADDLE_BINARY_DIR}/python/paddle/distributed/auto_parallel/proto/")
    add_custom_command(
      TARGET auto_parallel_py_proto
      POST_BUILD
      COMMAND
        ${CMAKE_COMMAND} -E make_directory
        ${PADDLE_BINARY_DIR}/python/paddle/distributed/auto_parallel/proto/
      COMMAND
        ${CMAKE_COMMAND} -E touch
        ${PADDLE_BINARY_DIR}/python/paddle/distributed/auto_parallel/proto/__init__.py
      COMMAND copy /Y auto_parallel_*.py ${auto_parallel_proto_dstpath}
      COMMENT
        "Copy generated python proto into directory paddle/distributed/auto_parallel/proto."
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  endif()
endif()

collect_srcs(
  core_srcs
  SRCS
  device_mesh.cc
  process_mesh.cc
  dist_attr.cc
  dist_mapper.cc
  dist_tensor.cc
  dist_meta_tensor.cc
  proto_helper.cc
  placement_types.cc
  inferspmd_utils.cc)

add_subdirectory(reshard)
