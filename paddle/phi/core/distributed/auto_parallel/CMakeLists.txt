proto_library(auto_parallel_proto SRCS auto_parallel.proto)

set(DISTRIBUTED_SRCS "")

if(WITH_DISTRIBUTE)
  list(
    APPEND
    DISTRIBUTED_SRCS
    dist_tensor.cc
    reshard_function.cc
    reshard_split_functor.cc
    reshard_utils.cc
    r_to_s_reshard_function.cc)
endif()

collect_srcs(
  core_srcs
  SRCS
  device_mesh.cc
  process_mesh.cc
  dist_attr.cc
  dist_mapper.cc
  ${DISTRIBUTED_SRCS})
