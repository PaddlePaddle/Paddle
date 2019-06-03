# if(LITE_WITH_LIGHT_WEIGHT_FRAMEWORK)
#     set(utils_DEPS)
#     lite_cc_test(test_logging_lite SRCS logging_test.cc)
# else()
# endif()

set(utils_DEPS glog)

lite_cc_test(test_varient SRCS varient_test.cc DEPS utils_lite)
cc_library(any_lite SRCS any.cc)
cc_library(utils_lite SRCS cp_logging.cc DEPS ${utils_DEPS} any_lite)
cc_library(blob_map_lite SRCS blob_map.cc)

lite_cc_test(test_blob_map_lite SRCS blob_map_test.cc DEPS blob_map_lite)
