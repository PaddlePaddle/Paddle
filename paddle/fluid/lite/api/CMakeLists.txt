cc_library(cxx_api_lite SRCS cxx_api.h DEPS scope_lite executor_lite host_kernels ops_lite)

cc_test(test_cxx_api_lite SRCS cxx_api_test.cc DEPS cxx_api_lite model_parser_lite)
