if(WITH_CINN)
  paddle_test(share_varinfo_into_cinn_pass_test SRCS
              share_varinfo_into_cinn_pass_test.cc)
  list(APPEND EAGER_DELETETION_PASS_DEPS share_varinfo_into_cinn_pass)
endif()

paddle_test(test_reference_count_pass_last_lived_ops SRCS
            test_reference_count_pass_last_lived_ops.cc DEPS common)
