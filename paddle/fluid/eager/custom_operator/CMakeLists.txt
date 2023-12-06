cc_library(
  custom_operator_node
  SRCS custom_operator_node.cc
  DEPS phi common grad_node_info custom_operator utils custom_operator_utils)

cc_library(
  custom_operator_utils
  SRCS custom_operator_utils.cc
  DEPS phi common grad_node_info custom_operator utils)
