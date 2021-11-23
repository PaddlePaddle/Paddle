cc_test(test_egr_task_eager_utils SRCS eager_utils_test.cc DEPS ${eager_deps} grad_node_info autograd_meta)
