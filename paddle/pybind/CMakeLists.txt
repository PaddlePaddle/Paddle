if(WITH_PYTHON)
cc_library(paddle_pybind SHARED
    SRCS pybind.cc
    DEPS pybind python backward
    ${OP_LIBRARY})
endif(WITH_PYTHON)
