if(WITH_PYTHON)
  cc_library(proto_bind SRCS protobuf.cc)
  cc_library(paddle_pybind SHARED
    SRCS pybind.cc
    DEPS proto_bind pybind python backward
    ${GLOB_OP_LIB})
endif(WITH_PYTHON)
