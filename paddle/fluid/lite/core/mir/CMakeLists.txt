cc_library(mir_node SRCS node.cc)
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
        DEPS mir_pass types_lite)

cc_test(test_mir_pass_manager SRCS pass_manager_test.cc DEPS mir_pass_manager mir_passes)
cc_test(test_ssa_graph SRCS ssa_graph_test.cc DEPS
        mir_ssa_graph scope_lite op_lite
        ops_lite
        host_kernels
        mir_passes
        mir_pass_manager
        program_fake_utils
        )
cc_test(test_variable_place_infrence_pass SRCS variable_place_inference_pass_test.cc DEPS
        ops_lite
        host_kernels
        kernels_cuda
        mir_passes
        mir_pass_manager
        optimizer_lite
        program_fake_utils
        target_wrapper_host
        target_wrapper_cuda
        )
