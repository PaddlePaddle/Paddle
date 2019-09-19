cc_library(fusion_group_pass
    SRCS fusion_group_pass.cc elementwise_group_detector.cc
    DEPS graph_pattern_detector pass)
cc_test(test_fusion_group_pass SRCS fusion_group_pass_tester.cc DEPS fusion_group_pass)
