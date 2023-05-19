cc_library(
  op_dist_attr
  SRCS dist_attr.cc
  DEPS phi auto_parallel_proto proto_desc)

add_subdirectory(test)

