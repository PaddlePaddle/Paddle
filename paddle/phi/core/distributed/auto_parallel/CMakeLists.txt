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
  reshard_function.cc
  reshard_split_functor.cc
  reshard_concat_functor.cc
  reshard_all_gather_functor.cc
  r_to_s_reshard_function.cc
  s_to_r_reshard_function.cc)
