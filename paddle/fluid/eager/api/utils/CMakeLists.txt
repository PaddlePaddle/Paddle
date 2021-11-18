cc_library(global_utils SRCS global_utils.cc DEPS enforce)
cc_library(hook_utils SRCS hook_utils.cc DEPS function_api pten autograd_meta grad_node_info utils accumulation_node)
