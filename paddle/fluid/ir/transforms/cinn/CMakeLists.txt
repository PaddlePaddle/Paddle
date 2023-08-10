# cc_library(
#   tensor_node
#   SRCS tensor_node.cc
#   DEPS ir)

# cc_library(
#   op_node
#   SRCS op_node.cc
#   DEPS ir)

# cc_library(
#   pd_op_to_kernel_pass
#   SRCS pd_op_to_kernel_pass.cc
#   DEPS phi_utils pd_interface pd_trait ir)

# cc_library(
#   _constant_folding_pass
#   SRCS constant_folding_pass.cc
#   DEPS standalone_executor phi pd_op_to_kernel_pass transform_general_functions
#        ir)

# cc_library(
#   fusion_merge_pass
#   SRCS fusion_merge_pass.cc
#   DEPS phi_utils pd_interface pd_trait ir)
