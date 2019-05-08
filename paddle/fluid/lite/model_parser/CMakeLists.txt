cc_library(runtime_lite SRCS runtime.cc)
lite_cc_test(test_model_parser_lite SRCS model_parser_test.cc DEPS model_parser_lite)
if(LITE_WITH_LIGHT_WEIGHT_FRAMEWORK)
    cc_library(compatible_pb_lite SRCS compatible_pb.cc DEPS op_desc_lite var_desc_lite)
else()
    cc_library(compatible_pb_lite SRCS compatible_pb.cc DEPS framework_proto proto_desc)
endif(LITE_WITH_LIGHT_WEIGHT_FRAMEWORK)

set(model_parser_deps variable_lite scope_lite ${tensor_lite} scope_lite
                      target_wrapper_host
                      compatible_pb_lite
                      )
if (LITE_WITH_CUDA)
    set(model_parser_deps ${model_parser_deps} target_wrapper_cuda)
endif()
cc_library(model_parser_lite SRCS model_parser.cc DEPS ${model_parser_deps})

add_subdirectory(pb)
