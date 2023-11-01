cc_library(
  custom_operator_node
  SRCS custom_operator_node.cc
  DEPS phi grad_node_info custom_operator utils custom_operator_run_kernel_impl)

cc_library(
  custom_operator_run_kernel_impl
  SRCS custom_operator_run_kernel_impl.cc
  DEPS phi grad_node_info custom_operator utils)
