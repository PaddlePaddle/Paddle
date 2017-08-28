if(WITH_PYTHON)
cc_library(paddle_pybind SHARED
    SRCS pybind.cc
    DEPS pybind python backward
    sgd_op
    gather_op
    scatter_op
    add_op
    mul_op
    rowwise_add_op
    sigmoid_op
    softmax_op
    mean_op
    cross_entropy_op
    recurrent_op
    uniform_random_op
    gaussian_random_op
    fill_zeros_like_op
    lookup_table_op
    scale_op
    minus_op)
endif(WITH_PYTHON)
