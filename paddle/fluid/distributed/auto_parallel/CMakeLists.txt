proto_library(auto_parallel_proto SRCS auto_parallel.proto)

cc_library(
  device_mesh
  SRCS device_mesh.cc
  DEPS auto_parallel_proto phi_enforce)

cc_library(
  process_mesh
  SRCS process_mesh.cc
  DEPS auto_parallel_proto phi_enforce)

cc_library(
  dist_attr
  SRCS dist_attr.cc
  DEPS process_mesh auto_parallel_proto proto_desc phi_enforce)

cc_library(
  dist_mapper
  SRCS dist_mapper.cc
  DEPS device_mesh auto_parallel_proto phi_enforce)

cc_library(auto_parallel DEPS device_mesh process_mesh dist_attr dist_mapper)
