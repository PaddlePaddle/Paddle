cc_library(tensor_utils SRCS tensor_utils.cc DEPS pten pten_api autograd_meta grad_node_info accumulation_node)
cc_library(hook_utils SRCS hook_utils.cc DEPS pten tensor_utils autograd_meta grad_node_info utils accumulation_node)
cc_library(global_utils SRCS global_utils.cc DEPS place)
