set(eager_deps eager_api utils function_api backward autograd_meta grad_node_info tcmpt)
set(fluid_deps tracer layer proto_desc operator op_registry variable_helper scale_op memcpy)

add_subdirectory(benchmark)
add_subdirectory(generated)

cc_test(test_egr_utils SRCS eager_utils_test.cc DEPS ${eager_deps} ${fluid_deps})
cc_test(test_egr_autocodegen SRCS generated_test.cc DEPS ${eager_deps} ${fluid_deps} sigmoid_function sigmoid_node)

cc_test(test_egr_backward SRCS backward_test.cc DEPS ${eager_deps} ${fluid_deps})
cc_test(test_egr_forward SRCS forward_test.cc DEPS ${eager_deps} ${fluid_deps})
cc_test(test_egr_hook SRCS hook_test.cc DEPS ${eager_deps} ${fluid_deps})
cc_test(test_egr_cross_batch_accumulation SRCS cross_batch_accumulation_test.cc DEPS ${eager_deps} ${fluid_deps})
cc_test(test_egr_fwd_bwd_joint SRCS fwd_bwd_joint_test.cc DEPS ${eager_deps} ${fluid_deps})
