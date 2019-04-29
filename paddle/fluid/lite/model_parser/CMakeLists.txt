cc_library(model_parser_lite SRCS model_parser.cc DEPS variable_lite scope_lite tensor_lite scope_lite)
cc_library(runtime_lite SRCS runtime.cc)
cc_test(test_model_parser_lite SRCS model_parser_test.cc DEPS model_parser_lite)
if(LITE_WITH_LIGHT_WEIGHT_FRAMEWORK)
    cc_library(compatible_pb_lite SRCS compatible_pb.cc DEPS op_desc_lite var_desc_lite)
else()
    cc_library(compatible_pb_lite SRCS compatible_pb.cc DEPS framework_proto)
endif(LITE_WITH_LIGHT_WEIGHT_FRAMEWORK)


add_subdirectory(pb)
