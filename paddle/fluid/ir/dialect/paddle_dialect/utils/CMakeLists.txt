cc_library(op_yaml_info_parser NOT_FOR_INFER SRCS op_yaml_info_parser.cc)
cc_library(
  pd_dialect_utils NOT_FOR_INFER
  SRCS utils.cc
  DEPS pd_dialect_core)
