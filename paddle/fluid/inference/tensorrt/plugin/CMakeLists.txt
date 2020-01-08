nv_library(tensorrt_plugin
           SRCS trt_plugin.cc split_op_plugin.cu elementwise_op_plugin.cu
           prelu_op_plugin.cu  trt_plugin_factory.cc gelu_op_plugin.cu
           pool_op_plugin.cu swish_op_plugin.cu layer_norm_op_plugin.cu instance_norm_op_plugin.cu
           DEPS enforce tensorrt_engine prelu tensor)
