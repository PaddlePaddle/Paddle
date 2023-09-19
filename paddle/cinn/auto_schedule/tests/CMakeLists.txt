if(WITH_CUDA AND (NOT WITH_CUDNN))
  cinn_cc_test(
    test_performance_comparison
    ARGS
    "--resnet50_model_dir=${THIRD_PARTY_PATH}/ResNet50"
    SRCS
    performance_comparison_test.cc
    DEPS
    cinncore
    test_program_builder)
endif()
