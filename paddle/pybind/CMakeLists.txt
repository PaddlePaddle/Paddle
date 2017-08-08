cc_library(paddle_pybind SHARED
    SRCS pybind.cc
    DEPS pybind python backward
	fc_op
	sgd_op
	add_op
	mean_op
  guassian_random_op
	cross_entropy_op
	recurrent_op
	fill_zeros_like_op)
