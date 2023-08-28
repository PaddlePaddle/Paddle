cc_library(
  fusion_merge_pass
  SRCS fusion_merge_pass.cc
  DEPS phi_utils pd_interface pd_trait pd_dialect_core)

cc_library(
  general_fusion_merge_pass
  SRCS general_fusion_merge_pass.cc
  DEPS phi_utils pd_interface fusion_merge_pass pd_trait pd_dialect_core)
