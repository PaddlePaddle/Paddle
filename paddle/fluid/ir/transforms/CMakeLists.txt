cc_library(
  transform_general_functions
  SRCS transform_general_functions.cc
  DEPS pd_dialect_core)

cc_library(
  pd_op_to_kernel_pass
  SRCS pd_op_to_kernel_pass.cc
  DEPS pd_kernel_dialect pd_dialect_core pd_dialect_utils)

cc_library(
  _constant_folding_pass
  SRCS constant_folding_pass.cc
  DEPS standalone_executor pd_op_to_kernel_pass transform_general_functions)
