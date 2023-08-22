add_subdirectory(cinn)

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

cc_library(
  fusion_merge_pass
  SRCS fusion_merge_pass.cc
  DEPS phi_utils pd_interface pd_trait pd_dialect_core)

cc_library(
  general_fusion_merge_pass
  SRCS general_fusion_merge_pass.cc
  DEPS phi_utils pd_interface fusion_merge_pass pd_trait pd_dialect_core)
