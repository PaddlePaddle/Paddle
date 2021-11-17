cc_library(function_api SRCS function_api.cc DEPS pten pten_api layer proto_desc operator op_registry variable_helper memcpy scale_op)
cc_library(eager_scale SRCS scale.cc DEPS function_api pten autograd_meta scale_node)
