cc_library(code_generator SRCS code_generator.cc code_generator_helper.cc DEPS graph)
if(NOT APPLE AND NOT WIN32)
    if(WITH_GPU)
        cc_test(test_code_generator SRCS code_generator_tester.cc DEPS code_generator device_code lod_tensor)
    endif()
endif()

cc_library(fusion_group_pass
    SRCS fusion_group_pass.cc elementwise_group_detector.cc
    DEPS graph_pattern_detector pass)
cc_test(test_fusion_group_pass SRCS fusion_group_pass_tester.cc DEPS fusion_group_pass graph_viz_pass)
