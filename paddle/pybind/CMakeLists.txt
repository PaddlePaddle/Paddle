if(WITH_PYTHON)
  cc_library(paddle_pybind SHARED
    SRCS pybind.cc exception.cc protobuf.cc
    DEPS pybind python backward proto_desc tensor_array paddle_memory executor
    ${GLOB_OP_LIB})
endif(WITH_PYTHON)
