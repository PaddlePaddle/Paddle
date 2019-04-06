cc_library(fc_compute_host SRCS fc_compute.cc DEPS tensor_lite)
cc_library(relu_compute_host SRCS relu_compute.cc DEPS tensor_lite)

cc_test(test_fc_compute SRCS fc_compute_test.cc DEPS fc_compute_host fc_op_lite)
