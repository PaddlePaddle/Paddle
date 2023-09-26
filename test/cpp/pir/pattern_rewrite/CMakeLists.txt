paddle_test(pattern_rewrite_test SRCS pattern_rewrite_test.cc DEPS pir)

set_tests_properties(
  pattern_rewrite_test PROPERTIES ENVIRONMENT
                                  "FLAGS_enable_new_ir_in_executor=true")
