cc_library(
  accumulation_node
  SRCS accumulation_node.cc
  DEPS gradient_accumulator grad_node_info)
