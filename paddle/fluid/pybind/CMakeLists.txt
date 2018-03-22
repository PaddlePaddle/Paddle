if(WITH_PYTHON)
  if(WITH_AMD_GPU)
    hip_library(paddle_pybind SHARED
      SRCS pybind.cc exception.cc protobuf.cc const_value.cc recordio.cc
      DEPS pybind python backward proto_desc paddle_memory executor prune init profiler feed_fetch_method
      ${GLOB_OP_LIB})
  else()
    cc_library(paddle_pybind SHARED
      SRCS pybind.cc exception.cc protobuf.cc const_value.cc recordio.cc
      DEPS pybind python backward proto_desc paddle_memory executor prune init profiler feed_fetch_method
      ${GLOB_OP_LIB})
    if(NOT APPLE AND NOT ANDROID)
      target_link_libraries(paddle_pybind rt)
    endif(NOT APPLE AND NOT ANDROID)
  endif(WITH_AMD_GPU)
endif(WITH_PYTHON)
