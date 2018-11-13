nv_library(tensorrt_plugin SRCS plugin_factory.cc plugin_utils.cc
trt_plugin.cc split_op_plugin.cu DEPS enforce)
