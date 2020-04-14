nv_library(tensorrt_plugin
           SRCS trt_plugin.cc split_op_plugin.cu elementwise_op_plugin.cu
           prelu_op_plugin.cu  trt_plugin_factory.cc gelu_op_plugin.cu 
           pool_op_plugin.cu swish_op_plugin.cu layer_norm_op_plugin.cu
instance_norm_op_plugin.cu emb_eltwise_layernorm_plugin.cu
qkv_to_context_plugin.cu skip_layernorm_op_plugin.cu slice_op_plugin.cu hard_swish_op_plugin.cu
           DEPS enforce tensorrt_engine prelu tensor bert_encoder_functor) 
