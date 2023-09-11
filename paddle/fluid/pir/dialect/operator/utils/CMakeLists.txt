cc_library(op_yaml_info_parser SRCS op_yaml_info_parser.cc)
cc_library(
  pd_op_dialect_utils
  SRCS utils.cc
  DEPS pd_op_dialect_core)
