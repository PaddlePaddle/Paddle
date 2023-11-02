file(GLOB FUSION_PASS_SRCS "fusion/*.cc")

cc_library(
  fusion_passes
  SRCS ${FUSION_PASS_SRCS}
  DEPS drr)

cc_library(
  transform_general_functions
  SRCS transform_general_functions.cc
  DEPS dialect)

cc_library(
  pd_op_to_kernel_pass
  SRCS pd_op_to_kernel_pass.cc
  DEPS dialect)

cc_library(
  pd_constant_folding_pass
  SRCS constant_folding_pass.cc
  DEPS standalone_executor pd_op_to_kernel_pass transform_general_functions)

cc_library(
  pd_inplace_pass
  SRCS inplace_pass.cc
  DEPS dialect)

if(WITH_CINN)
  cc_library(
    pd_build_cinn_pass
    SRCS build_cinn_pass.cc
    DEPS dialect cinn_op_dialect pir cinnapi)
endif()
