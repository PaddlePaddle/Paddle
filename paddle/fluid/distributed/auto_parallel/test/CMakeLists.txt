cc_test(
  device_mesh_test
  SRCS device_mesh_test.cc
  DEPS phi common)

cc_test(
  process_mesh_test
  SRCS process_mesh_test.cc
  DEPS phi common)

cc_test(
  dist_attr_test
  SRCS dist_attr_test.cc
  DEPS phi common proto_desc)

cc_test(
  dist_mapper_test
  SRCS dist_mapper_test.cc
  DEPS phi common)

cc_test(
  spmd_rule_test
  SRCS spmd_rule_test.cc
  DEPS spmd_rules)
