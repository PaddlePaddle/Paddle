cc_test(test_egr_ds_eager_tensor SRCS eager_tensor_test.cc DEPS ${eager_deps} )
cc_test(test_egr_ds_auotgrad_meta SRCS autograd_meta_test.cc DEPS ${eager_deps} grad_node_info)
cc_test(test_egr_ds_grad_node_info SRCS grad_node_info_test.cc DEPS ${eager_deps} grad_node_info)
