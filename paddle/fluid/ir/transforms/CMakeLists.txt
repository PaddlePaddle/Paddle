cc_library(
  transform_general_functions
  SRCS transform_general_functions.cc
  DEPS ir phi pd_dialect)

cc_library(
  pd_op_to_kernel_pass
  SRCS pd_op_to_kernel_pass.cc
  DEPS ir phi_utils pd_interface)
