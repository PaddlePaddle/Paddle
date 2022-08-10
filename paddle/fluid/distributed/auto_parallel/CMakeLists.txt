proto_library(auto_parallel_proto SRCS auto_parallel.proto)

cc_library(
  device_mesh
  SRCS device_mesh.cc
  DEPS auto_parallel_proto)
cc_test(
  device_mesh_test
  SRCS device_mesh_test.cc
  DEPS device_mesh)

cc_library(
  process_mesh
  SRCS process_mesh.cc
  DEPS auto_parallel_proto)
cc_test(
  process_mesh_test
  SRCS process_mesh_test.cc
  DEPS process_mesh)

cc_library(
  dist_attr
  SRCS dist_attr.cc
  DEPS process_mesh auto_parallel_proto proto_desc)
cc_test(
  dist_attr_test
  SRCS dist_attr_test.cc
  DEPS dist_attr)

cc_library(
  dist_mapper
  SRCS dist_mapper.cc
  DEPS device_mesh auto_parallel_proto)
cc_test(
  dist_mapper_test
  SRCS dist_mapper_test.cc
  DEPS dist_mapper)

cc_library(
  auto_parallel
  SRCS device_mesh.cc
  DEPS process_mesh device_mesh dist_attr dist_mapper)
