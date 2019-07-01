lite_cc_library(mir_node SRCS node.cc DEPS framework_proto_lite)
lite_cc_library(mir_ssa_graph SRCS ssa_graph.cc DEPS mir_node program_lite)
lite_cc_library(mir_pass SRCS pass.cc DEPS mir_ssa_graph)
lite_cc_library(mir_pass_manager SRCS pass_manager.cc DEPS mir_pass mir_ssa_graph mir_passes)
lite_cc_library(mir_pass_registry SRCS pass_registry.cc DEPS mir_pass_manager)

add_subdirectory(fusion)
add_subdirectory(elimination)

lite_cc_library(mir_passes
  SRCS
      fusion/fc_fuse_pass.cc
      fusion/conv_elementwise_add_activation_fuse_pass.cc
      fusion/conv_bn_fuse_pass.cc
      fusion/elementwise_add_activation_fuse_pass.cc
      fusion/quant_dequant_fuse_pass.cc
      elimination/identity_scale_eliminate_pass.cc
      static_kernel_pick_pass.cc
      variable_place_inference_pass.cc
      type_target_cast_pass.cc
      type_precision_cast_pass.cc
      io_copy_kernel_pick_pass.cc
      graph_visualize_pass.cc
      generate_program_pass.cc
      argument_type_display_pass.cc
      trans_weigths_pass.cc
      demo_pass.cc
      runtime_context_assign_pass.cc
  DEPS mir_pass types_lite context_lite ${mir_fusers})

# lite_cc_test(test_ssa_graph SRCS ssa_graph_test.cc DEPS
        #mir_ssa_graph scope_lite op_lite
        #fc_op_lite
        #${host_kernels}
        #mir_passes
        #mir_pass_manager
        #program_fake_utils
        #)
# lite_cc_test(test_variable_place_infrence_pass SRCS variable_place_inference_pass_test.cc
#   DEPS
#       mul_op_lite
#       feed_op_lite
#       fetch_op_lite
#       io_copy_op_lite
#       ${host_kernels}
#       mir_passes
#       mir_pass_manager
#       optimizer_lite
#       program_fake_utils
#       target_wrapper_host
#   PROFILE_DEPS basic_profiler_lite
#   CUDA_DEPS target_wrapper_cuda kernels_cuda
#   ARM_DEPS mul_compute_arm
#   X86_DEPS mul_compute_x86
# )

set(pattern_deps mir_node mir_ssa_graph op_lite)
if (WITH_TESTING)
  list(APPEND pattern_deps gtest)
endif()
lite_cc_library(pattern_matcher_lite SRCS pattern_matcher.cc DEPS ${pattern_deps})
lite_cc_test(test_pattern_matcher_lite SRCS pattern_matcher_test.cc DEPS pattern_matcher_lite)

lite_cc_library(pattern_matcher_high_api SRCS pattern_matcher_high_api.cc DEPS pattern_matcher_lite)


# for mobile, unnecessary to compile the following testings.
if (LITE_WITH_LIGHT_WEIGHT_FRAMEWORK)
    return()
endif()
lite_cc_test(test_mir_pass_manager SRCS pass_manager_test.cc DEPS mir_pass_manager mir_passes)


# TODO(wz) replace framework/proto to lite proto.
if (NOT LITE_WITH_LIGHT_WEIGHT_FRAMEWORK)
    # it depends on the fluid/framework/proto, that is too heavy for mobile execution.
    lite_cc_test(test_pattern_matcher_high_api SRCS pattern_matcher_high_api_test.cc DEPS
        pattern_matcher_high_api proto_desc mir_pass_manager fc_op_lite mul_op_lite elementwise_ops_lite
        mir_passes compatible_pb_lite program_lite ${ops_lite})
endif()

message(STATUS "----> Ops lite: ${ops_lite}")
message(STATUS "----> Host kernels: ${host_kernels}")
message(STATUS "----> X86 kernels: ${x86_kernels}")

lite_cc_test(test_lite_fc_fuse SRCS fusion/fc_fuse_pass_test.cc
   DEPS cxx_api_lite mir_passes
   ${ops_lite} ${host_kernels} ${x86_kernels} ${arm_kernels}
   ARGS --model_dir=${LITE_MODEL_DIR}/lite_fc_model
        --optimized_model=${LITE_MODEL_DIR}/lite_fc_model_opt SERIAL)

lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL} "lite_fc_model.tar.gz")
add_dependencies(test_lite_fc_fuse extern_lite_download_lite_fc_model_tar_gz)


lite_cc_test(test_lite_conv_elementwise_add_activation_fuse
             SRCS fusion/conv_elementwise_add_activation_fuse_pass_test.cc
             DEPS cxx_api_lite mir_passes
             ${ops_lite} ${host_kernels} ${x86_kernels})
lite_cc_test(test_lite_elementwise_add_activation_fuse
             SRCS fusion/elementwise_add_activation_fuse_pass_test.cc
             DEPS cxx_api_lite mir_passes
             ${ops_lite} ${host_kernels} ${x86_kernels})
