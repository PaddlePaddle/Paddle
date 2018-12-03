
set(PYBIND_DEPS pybind python proto_desc memory executor async_executor prune feed_fetch_method pass_builder parallel_executor profiler)
set(PYBIND_SRCS pybind.cc exception.cc protobuf.cc const_value.cc recordio.cc async_executor_py.cc)
if(WITH_PYTHON)
  if(WITH_AMD_GPU)
    hip_library(paddle_pybind SHARED
      SRCS ${PYBIND_SRCS}
      DEPS ARCHIVE_START ${PYBIND_DEPS}
      ${GLOB_OP_LIB} ${GLOB_OPERATOR_DEPS} ARCHIVE_END)
  else()
    cc_library(paddle_pybind SHARED
      SRCS ${PYBIND_SRCS}
      DEPS ${PYBIND_DEPS}
      ${GLOB_OP_LIB} ${GLOB_OPERATOR_DEPS})
    if(NOT APPLE AND NOT ANDROID AND NOT WIN32)
      target_link_libraries(paddle_pybind rt)
    endif(NOT APPLE AND NOT ANDROID AND NOT WIN32)
  endif(WITH_AMD_GPU)

  if(WIN32)
    if(WITH_GPU AND NOT WITH_DSO)
      get_property(cuda_modules GLOBAL PROPERTY CUDA_MODULES)
      target_link_libraries(paddle_pybind ${cuda_modules})
    endif(WITH_GPU AND NOT WITH_DSO)
    target_link_libraries(paddle_pybind shlwapi)
  endif(WIN32)

  cc_test(tensor_py_test SRCS tensor_py_test.cc DEPS python)
endif(WITH_PYTHON)
