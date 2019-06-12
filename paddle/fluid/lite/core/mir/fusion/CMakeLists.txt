cc_library(mir_fuse_passes
        SRCS fc_fuse_pass.cc
        DEPS mir_pass mir_pass_manager pattern_matcher_high_api)

message(STATUS "----> Ops lite: ${ops_lite}")
message(STATUS "----> Host kernels: ${host_kernels}")
message(STATUS "----> X86 kernels: ${x86_kernels}")

lite_cc_test(test_lite_fc_fuse SRCS fc_fuse_pass_test.cc
   DEPS cxx_api_lite model_parser_lite target_wrapper_host mir_fuse_passes
   ${ops_lite} ${host_kernels} ${x86_kernels}
   PROFILE_DEPS basic_profiler_lite
   ARGS --model_dir=${LITE_MODEL_DIR}/lite_fc_model
        --optimized_model=${LITE_MODEL_DIR}/lite_fc_model_opt SERIAL)

lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL} "lite_fc_model.tar.gz")
add_dependencies(test_lite_fc_fuse extern_lite_download_lite_fc_model_tar_gz)
