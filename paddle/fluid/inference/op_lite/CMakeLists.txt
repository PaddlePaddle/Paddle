cc_library(op_lite SRCS op_lite.cc)
cc_library(fc_op_lite SRCS fc_op.cc DEPS lod_tensor )

cc_library(lite_ops SRCS ops.cc DEPS op_lite fc_op)
