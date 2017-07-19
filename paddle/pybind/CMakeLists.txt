cc_library(paddle_pybind SHARED SRCS pybind.cc DEPS pybind python
        mul_op rowwise_add_op sigmoid_op softmax_op sgd_op
        add_op fc_op)
