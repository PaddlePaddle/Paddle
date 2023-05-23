cc_library(
  op_dist_attr
  SRCS dist_attr.cc
  DEPS dist_attr process_mesh dist_mapper auto_parallel_proto proto_desc
       phi_enforce)

add_subdirectory(test)
add_subdirectory(spmd_rules)

cc_library(auto_parallel DEPS device_mesh process_mesh dist_attr dist_mapper
                              dist_tensor_spec)
