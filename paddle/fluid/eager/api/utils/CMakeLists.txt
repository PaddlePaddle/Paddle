cc_library(global_utils SRCS global_utils.cc DEPS place)
cc_library(tensor_utils SRCS tensor_utils.cc DEPS pten pten_api autograd_meta grad_node_info accumulation_node)
