cc_library(
  code_generator
  SRCS operation.cc code_generator.cc code_generator_helper.cc
  DEPS graph subgraph_detector)
if(WITH_GPU OR WITH_ROCM)
  paddle_test(
    test_code_generator
    SRCS code_generator_tester.cc
    DEPS common)
endif()

cc_library(
  fusion_group_pass
  SRCS fusion_group_pass.cc elementwise_group_detector.cc
  DEPS subgraph_detector fuse_pass_base code_generator phi common)
paddle_test(
  test_fusion_group_pass
  SRCS fusion_group_pass_tester.cc)
if(WITH_TESTING AND TEST test_code_generator)
  set_tests_properties(test_code_generator PROPERTIES TIMEOUT 120)
endif()
