cc_library(paddle_pybind SHARED
    SRCS pybind.cc
    DEPS pybind python
	fc_op
	sgd_op
	add_op
	mean_op
	cross_entropy_op
	recurrent_network_op)
