cc_library(mir_fuse_passes
        SRCS fc_fuse_pass.cc
        DEPS mir_pass mir_pass_manager pattern_matcher_high_api)

cc_test(test_lite_fc_fuse SRCS fc_fuse_pass_test.cc
   DEPS cxx_api_lite model_parser_lite target_wrapper_host mir_fuse_passes
   ${ops_lite} ${host_kernels} ${x86_kernels}
   ARGS --model_dir=${LITE_MODEL_DIR}/lite_fc_model
        --optimized_model=${LITE_MODEL_DIR}/lite_fc_model_opt SERIAL)
