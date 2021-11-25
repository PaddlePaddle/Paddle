cc_library(tensor_utils SRCS tensor_utils.cc DEPS pten pten_api autograd_meta accumulation_node)
cc_library(global_utils SRCS global_utils.cc DEPS place)
