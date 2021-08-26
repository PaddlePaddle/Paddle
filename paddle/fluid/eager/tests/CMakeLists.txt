add_subdirectory(benchmark)

cc_test(backward_test SRCS backward_test.cc DEPS eager_api function_api backward autograd_meta grad_node_info top)
cc_test(forward_test SRCS forward_test.cc DEPS eager_api function_api backward autograd_meta grad_node_info top)
cc_test(hook_test SRCS hook_test.cc DEPS eager_api function_api backward autograd_meta grad_node_info top)
cc_test(cross_batch_accumulation_test SRCS cross_batch_accumulation_test.cc DEPS eager_api function_api backward autograd_meta grad_node_info top)
cc_test(fwd_bwd_joint_test SRCS fwd_bwd_joint_test.cc DEPS eager_api function_api backward autograd_meta grad_node_info top)
