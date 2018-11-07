
set(PYBIND_DEPS pybind python proto_desc memory executor prune
  feed_fetch_method pass_builder jemalloc)
set(PYBIND_SRCS pybind.cc exception.cc protobuf.cc const_value.cc)
if(NOT WIN32)
list(APPEND PYBIND_DEPS parallel_executor profiler)
list(APPEND PYBIND_SRCS recordio.cc)
endif()
if(WITH_PYTHON)
  if(WITH_AMD_GPU)
    hip_library(paddle_pybind SHARED
      SRCS ${PYBIND_SRCS}
      DEPS ${PYBIND_DEPS}
      ${GLOB_OP_LIB})
  else()
    cc_library(paddle_pybind SHARED
      SRCS ${PYBIND_SRCS}
      DEPS ${PYBIND_DEPS}
      ${GLOB_OP_LIB})
    if(NOT APPLE AND NOT ANDROID AND NOT WIN32)
      target_link_libraries(paddle_pybind rt)
      include_directories(${JEMALLOC_INSTALL_DIR}/include)
      link_directories(${JEMALLOC_INSTALL_DIR}/lib)
      target_link_libraries(paddle_pybind "-ljemalloc ${JEMALLOC_LIBS}")
    endif(NOT APPLE AND NOT ANDROID AND NOT WIN32)
  endif(WITH_AMD_GPU)

  cc_test(tensor_py_test SRCS tensor_py_test.cc DEPS python)
endif(WITH_PYTHON)
