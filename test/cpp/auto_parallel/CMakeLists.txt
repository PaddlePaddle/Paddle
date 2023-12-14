cc_test(device_mesh_test SRCS device_mesh_test.cc)

cc_test(process_mesh_test SRCS process_mesh_test.cc)

cc_test(
  dist_attr_test
  SRCS dist_attr_test.cc
  DEPS proto_desc)

if(WITH_DISTRIBUTE)
  cc_library(spmd_rule_test_util SRCS spmd_rule_test_util.cc)
  add_dependencies(spmd_rule_test_util spmd_rules)
  cc_test(
    dist_tensor_test
    SRCS dist_tensor_test.cc
    DEPS phi common)

  paddle_test(
    spmd_rule_test
    SRCS
    spmd_rule_test.cc
    DEPS
    spmd_rule_test_util
    spmd_rules
    phi
    common)
  paddle_test(
    softmax_grad_spmd_rule_test
    SRCS
    softmax_grad_spmd_rule_test.cc
    DEPS
    spmd_rule_test_util
    spmd_rules
    phi)

endif()

cc_test(
  dist_mapper_test
  SRCS dist_mapper_test.cc
  DEPS phi)
