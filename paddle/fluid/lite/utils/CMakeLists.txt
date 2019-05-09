if(LITE_WITH_LIGHT_WEIGHT_FRAMEWORK)
    set(utils_DEPS)
else()
    set(utils_DEPS glog)
endif()

lite_cc_test(test_logging_lite SRCS logging_test.cc)

cc_library(utils_lite SRCS cp_logging.cc DEPS ${utils_DEPS})
lite_cc_test(test_varient SRCS varient_test.cc DEPS utils_lite)

cc_library(any_lite SRCS any.cc)
