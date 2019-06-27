#lite_cc_library(runtime_lite SRCS runtime.cc)

#TODO(Superjomn) enable it again.
if(NOT LITE_ON_MOBILE)
    lite_cc_test(test_model_parser_lite SRCS model_parser_test.cc
      DEPS model_parser_lite framework_proto_lite
      ARGS --model_dir=${LITE_MODEL_DIR}/lite_naive_model)
    if(WITH_TESTING)
        add_dependencies(test_model_parser_lite extern_lite_download_lite_naive_model_tar_gz)
    endif(WITH_TESTING)
endif()

lite_cc_library(compatible_pb_lite SRCS compatible_pb.cc
  DEPS op_desc_lite framework_proto_lite var_desc_lite cpp_op_desc_lite)

lite_cc_library(model_parser_lite SRCS model_parser.cc DEPS
    variable_lite scope_lite ${tensor_lite} scope_lite
    target_wrapper_host
    compatible_pb_lite
    memory_lite
    CUDA_DEPS target_wrapper_cuda)

lite_cc_test(test_op_desc_lite SRCS op_desc_test.cc DEPS cpp_op_desc_lite op_desc_lite compatible_pb_lite)

add_subdirectory(pb)
add_subdirectory(cpp)
