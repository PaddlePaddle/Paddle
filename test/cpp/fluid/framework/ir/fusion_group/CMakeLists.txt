if(WITH_GPU OR WITH_ROCM)
  paddle_test(test_code_generator SRCS code_generator_tester.cc DEPS common)
endif()
if(WITH_TESTING AND TEST test_code_generator)
  set_tests_properties(test_code_generator PROPERTIES TIMEOUT 120)
endif()

paddle_test(test_fusion_group_pass SRCS fusion_group_pass_tester.cc)
