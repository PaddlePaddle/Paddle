cc_library(
  op_dist_attr
  SRCS dist_attr.cc
  DEPS dist_attr process_mesh dist_mapper auto_parallel_proto proto_desc
       phi_enforce)

add_subdirectory(test)
