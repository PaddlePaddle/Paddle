cc_test(test_egr_task_eager_utils SRCS eager_utils_test.cc DEPS ${eager_deps})
cc_test(test_egr_task_forward_autograd SRCS forward_autograd_test.cc DEPS ${eager_deps} ${fluid_deps} eager_scale scale_node)
