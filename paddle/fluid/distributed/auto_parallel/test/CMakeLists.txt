cc_test(
  device_mesh_test
  SRCS device_mesh_test.cc
  DEPS device_mesh)

cc_test(
  process_mesh_test
  SRCS process_mesh_test.cc
  DEPS process_mesh)

cc_test(
  dist_attr_test
  SRCS dist_attr_test.cc
  DEPS dist_attr)

cc_test(
  dist_mapper_test
  SRCS dist_mapper_test.cc
  DEPS dist_mapper)
