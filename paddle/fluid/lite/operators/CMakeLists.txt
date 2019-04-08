cc_library(fc_op_lite SRCS fc_op.cc DEPS op_lite op_params_lite tensor_lite proto_desc)
cc_library(relu_op_lite SRCS relu_op.cc DEPS op_lite)
cc_library(op_params_lite SRCS op_params.cc DEPS tensor_lite)
cc_library(ops_lite DEPS
        fc_op_lite
        relu_op_lite)

cc_test(test_fc_op_lite SRCS fc_op_test.cc DEPS fc_op_lite fc_compute_host)
