nv_library(tensorrt_plugin SRCS trt_plugin.cc split_op_plugin.cu
avg_pool_op_plugin.cu DEPS enforce pooling)
