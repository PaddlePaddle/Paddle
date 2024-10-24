cc_test(device_mesh_test SRCS device_mesh_test.cc)

cc_test(process_mesh_test SRCS process_mesh_test.cc)

cc_test(
  dist_attr_test
  SRCS dist_attr_test.cc
  DEPS proto_desc)

if(WITH_DISTRIBUTE)
  cc_library(
    spmd_rule_test_util
    SRCS spmd_rule_test_util.cc
    DEPS gtest)
  cc_test(
    dist_tensor_test
    SRCS dist_tensor_test.cc
    DEPS phi common)

  paddle_test(spmd_rule_test SRCS spmd_rule_test.cc DEPS spmd_rule_test_util
              phi)

  paddle_test(softmax_grad_spmd_rule_test SRCS softmax_grad_spmd_rule_test.cc
              DEPS spmd_rule_test_util phi)

  paddle_test(tile_spmd_rule_test SRCS tile_spmd_rule_test.cc DEPS
              spmd_rule_test_util phi)

  paddle_test(
    fused_linear_param_grad_add_spmd_rule_test SRCS
    fused_linear_param_grad_add_spmd_rule_test.cc DEPS spmd_rule_test_util phi)

  paddle_test(
    cross_entropy_softmax_spmd_rule_test SRCS
    cross_entropy_softmax_spmd_rule_test.cc DEPS spmd_rule_test_util phi)

  paddle_test(expand_as_spmd_rule_test SRCS expand_as_spmd_rule_test.cc DEPS
              spmd_rule_test_util phi)

  paddle_test(custom_op_spmd_rule_test SRCS custom_op_spmd_rule_test.cc DEPS
              spmd_rule_test_util phi)

  paddle_test(fused_rms_norm_spmd_rule_test SRCS
              fused_rms_norm_spmd_rule_test.cc DEPS spmd_rule_test_util phi)
  paddle_test(moe_gate_dispatch_spmd_rule_test SRCS
              moe_gate_dispatch_spmd_rule_test.cc DEPS spmd_rule_test_util phi)

  paddle_test(moe_combine_spmd_rule_test SRCS moe_combine_spmd_rule_test.cc
              DEPS spmd_rule_test_util phi)

endif()

cc_test(
  dist_mapper_test
  SRCS dist_mapper_test.cc
  DEPS phi)
