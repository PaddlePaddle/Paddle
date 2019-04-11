cc_library(model_parser_lite SRCS model_parser.cc DEPS variable_lite scope_lite tensor_lite scope_lite)
cc_library(runtime_lite SRCS runtime.cc)
cc_test(test_model_parser_lite SRCS model_parser_test.cc DEPS model_parser_lite)
