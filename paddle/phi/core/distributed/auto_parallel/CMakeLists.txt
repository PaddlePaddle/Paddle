proto_library(auto_parallel_proto SRCS auto_parallel.proto)

collect_srcs(
  core_srcs
  SRCS
  device_mesh.cc
  process_mesh.cc
  dist_attr.cc
  dist_mapper.cc
  reshard_utils.cc
  dist_tensor.cc
  dist_meta_tensor.cc
  inferspmd_utils.cc
  reshard_function.cc
  r_to_s_reshard_function.cc
  s_to_r_reshard_function.cc
  r_to_p_reshard_function.cc
  p_to_r_reshard_function.cc
  s_to_s_reshard_function.cc
  nd_mesh_reshard_function.cc
  same_status_reshard_function.cc)
