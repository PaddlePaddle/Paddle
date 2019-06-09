if (NOT LITE_WITH_PROFILE)
  return()
endif()

lite_cc_library(basic_profiler_lite SRCS basic_profiler.cc)
lite_cc_test(test_basic_profiler SRCS basic_profiler_test.cc DEPS basic_profiler_lite)
 
