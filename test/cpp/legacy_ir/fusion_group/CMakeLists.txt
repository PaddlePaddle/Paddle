# Fusion Group IR Pass Tests

paddle_test(test_fusion_group_pass SRCS fusion_group_pass_test.cc DEPS
            fusion_group_pass graph_viz_pass)

if(WITH_GPU OR WITH_ROCM)
  paddle_test(
    test_code_generator
    SRCS
    code_generator_test.cc
    DEPS
    code_generator
    phi
    common
    lod_tensor
    graph_viz_pass)
endif()
