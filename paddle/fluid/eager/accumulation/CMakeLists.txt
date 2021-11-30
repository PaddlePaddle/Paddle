cc_library(gradient_accumulation SRCS gradient_accumulation.cc DEPS blas pten pten_api var_type_traits layer math_function)
cc_library(accumulation_node SRCS accumulation_node.cc DEPS gradient_accumulation pten pten_api grad_node_info)
