cc_library(autograd_meta SRCS autograd_meta.cc DEPS top)
cc_library(grad_node_info SRCS grad_node_info.cc DEPS top autograd_meta)
