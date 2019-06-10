cc_library(mir_node SRCS node.cc DEPS framework_proto_lite)
cc_library(mir_ssa_graph SRCS ssa_graph.cc DEPS mir_node)
cc_library(mir_pass SRCS pass.cc DEPS mir_ssa_graph)
cc_library(mir_pass_manager SRCS pass_manager.cc DEPS mir_pass mir_ssa_graph mir_passes)
cc_library(mir_pass_registry SRCS pass_registry.cc DEPS mir_pass_manager)
cc_library(mir_passes
        SRCS static_kernel_pick_pass.cc
        variable_place_inference_pass.cc
        type_target_transform_pass.cc
        io_copy_kernel_pick_pass.cc
        graph_visualize_pass.cc
        generate_program_pass.cc
        argument_type_display_pass.cc
        demo_pass.cc
        runtime_context_assign_pass.cc
        DEPS mir_pass types_lite context_lite)

# for mobile, unnecessary to compile the following testings.
if (LITE_WITH_LIGHT_WEIGHT_FRAMEWORK)
    return()
endif()
cc_test(test_mir_pass_manager SRCS pass_manager_test.cc DEPS mir_pass_manager mir_passes)
cc_test(test_ssa_graph SRCS ssa_graph_test.cc DEPS
        mir_ssa_graph scope_lite op_lite
        fc_op_lite
        ${host_kernels}
        mir_passes
        mir_pass_manager
        program_fake_utils
        )
set(test_variable_place_infrence_pass_DEPS
        mul_op_lite
        feed_op_lite
        fetch_op_lite
        io_copy_op_lite
        ${host_kernels}
        mir_passes
        mir_pass_manager
        optimizer_lite
        program_fake_utils
        target_wrapper_host
        )
if (LITE_WITH_CUDA)
    set(test_variable_place_infrence_pass_DEPS
            ${test_variable_place_infrence_pass_DEPS} target_wrapper_cuda
            kernels_cuda
            )
endif()
cc_test(test_variable_place_infrence_pass SRCS variable_place_inference_pass_test.cc DEPS
        ${test_variable_place_infrence_pass_DEPS})

cc_library(pattern_matcher_lite SRCS pattern_matcher.cc DEPS mir_node mir_ssa_graph op_lite)
cc_test(test_pattern_matcher_lite SRCS pattern_matcher_tester.cc DEPS pattern_matcher_lite)

cc_library(pattern_matcher_high_api SRCS pattern_matcher_high_api.cc DEPS pattern_matcher_lite)
cc_test(test_pattern_matcher_high_api SRCS pattern_matcher_high_api_test.cc DEPS
    pattern_matcher_high_api proto_desc mir_pass_manager fc_op_lite mul_op_lite elementwise_ops_lite
    mir_passes compatible_pb_lite ${ops_lite})
