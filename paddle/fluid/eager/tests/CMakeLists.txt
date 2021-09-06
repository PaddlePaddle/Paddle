set(eager_deps eager_api function_api backward autograd_meta grad_node_info top)
set(fluid_deps tracer layer proto_desc operator op_registry variable_helper scale_op memcpy)

add_subdirectory(benchmark)

cc_test(backward_test SRCS backward_test.cc DEPS ${eager_deps} ${fluid_deps})
cc_test(forward_test SRCS forward_test.cc DEPS ${eager_deps} ${fluid_deps})
cc_test(hook_test SRCS hook_test.cc DEPS ${eager_deps} ${fluid_deps})
cc_test(cross_batch_accumulation_test SRCS cross_batch_accumulation_test.cc DEPS ${eager_deps} ${fluid_deps})
cc_test(fwd_bwd_joint_test SRCS fwd_bwd_joint_test.cc DEPS ${eager_deps} ${fluid_deps})
