if(WITH_PYTHON)
  if(WITH_AMD_GPU)
    hip_library(paddle_pybind SHARED
      SRCS pybind.cc exception.cc protobuf.cc const_value.cc recordio.cc
      DEPS pybind python proto_desc memory executor prune init profiler feed_fetch_method
           parallel_executor
      ${GLOB_OP_LIB})
  else()
    cc_library(paddle_pybind SHARED
      SRCS pybind.cc exception.cc protobuf.cc const_value.cc recordio.cc
      DEPS pybind python proto_desc memory executor prune init profiler feed_fetch_method
           parallel_executor
      ${GLOB_OP_LIB})
    if(NOT APPLE AND NOT ANDROID)
      target_link_libraries(paddle_pybind rt)
    endif(NOT APPLE AND NOT ANDROID)
  endif(WITH_AMD_GPU)

  cc_test(tensor_py_test SRCS tensor_py_test.cc DEPS python)
endif(WITH_PYTHON)
