cc_library(paddle_pybind SHARED SRCS pybind.cc DEPS pybind python
        add_op mul_op rowwise_add_op sigmoid_op softmax_op sgd_op)
