#cc_library(runtime_lite SRCS runtime.cc)

lite_cc_test(test_model_parser_lite SRCS model_parser_test.cc
  DEPS model_parser_lite framework_proto_lite
  ARGS --model_dir=${LITE_MODEL_DIR}/lite_naive_model)
if(WITH_TESTING)
add_dependencies(test_model_parser_lite extern_lite_download_lite_naive_model_tar_gz)
endif(WITH_TESTING)


if(LITE_WITH_LIGHT_WEIGHT_FRAMEWORK)
    cc_library(compatible_pb_lite SRCS compatible_pb.cc DEPS op_desc_lite var_desc_lite)
else()
    cc_library(compatible_pb_lite SRCS compatible_pb.cc DEPS framework_proto_lite proto_desc)
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
