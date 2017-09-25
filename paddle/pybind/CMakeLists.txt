if(WITH_PYTHON)
  cc_library(paddle_pybind SHARED
    SRCS pybind.cc
    DEPS pybind python backward paddle_memory
    ${GLOB_OP_LIB})
endif(WITH_PYTHON)
