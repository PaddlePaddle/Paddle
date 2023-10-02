core_gather_headers()

gather_srcs(
  cinnapi_src
  SRCS
  auto_gen_rule.cc
  auto_inline.cc
  auto_unroll.cc
  multi_level_tiling.cc
  skip_rule.cc
  auto_bind.cc)

if(WITH_TESTING)
  cinn_cc_library(
    auto_gen_rule_test_helper
    SRCS
    test_helper.cc
    DEPS
    glog
    gtest
    cinncore)
endif()

if(WITH_CUDA)
  cinn_nv_test(
    test_mix_rules
    SRCS
    mix_rules_test.cc
    DEPS
    cinncore
    auto_gen_rule_test_helper
    test_program_builder)
  cinn_nv_test(
    test_auto_bind
    SRCS
    auto_bind_test.cc
    DEPS
    cinncore
    auto_gen_rule_test_helper
    test_program_builder)
  cinn_nv_test(
    test_multi_level_tiling
    SRCS
    multi_level_tiling_test.cc
    DEPS
    cinncore
    auto_gen_rule_test_helper
    test_program_builder)
endif()

#cinn_cc_test(test_auto_inline SRCS auto_inline_test.cc DEPS cinncore auto_gen_rule_test_helper)
cinn_cc_test(test_skip_rule SRCS skip_rule_test.cc DEPS cinncore)
cinn_cc_test(test_auto_unroll SRCS auto_unroll_test.cc DEPS cinncore)
