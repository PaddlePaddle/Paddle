if(WITH_DISTRIBUTE)
  paddle_test(dist_dialect_test SRCS dist_dialect_test.cc)
  paddle_test(dist_dialect_reshard_pass_test SRCS dist_dialect_reshard_pass_test.cc)
endif()
