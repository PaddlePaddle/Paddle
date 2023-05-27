if(WITH_NEWIR)
  cc_test_old(pass_manager_test SRCS pass_manager_test.cc DEPS new_pass gtest)
endif()
