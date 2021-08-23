cc_library(eager_api SRCS fwd_ops_impl.cc api.cc DEPS function_helper top autograd_meta grad_node_info)
