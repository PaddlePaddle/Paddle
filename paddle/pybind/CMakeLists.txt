cc_library(paddle_pybind SHARED SRCS pybind.cc DEPS pybind python
        add_op fc_op cross_entropy_op)
