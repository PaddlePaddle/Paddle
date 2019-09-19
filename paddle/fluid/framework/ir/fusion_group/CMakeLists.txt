cc_library(fusion_group_pass
    SRCS fusion_group_pass.cc elementwise_pattern.cc
    DEPS graph_pattern_detector fuse_pass_base)
cc_test(test_fusion_group_pass SRCS fusion_group_pass_tester.cc DEPS fusion_group_pass)
