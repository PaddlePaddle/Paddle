add_subdirectory(spmd_rules)

cc_library(
  op_dist_attr
  SRCS dist_attr.cc
  DEPS phi auto_parallel_proto proto_desc)

cc_library(auto_parallel DEPS op_dist_attr spmd_rules)
