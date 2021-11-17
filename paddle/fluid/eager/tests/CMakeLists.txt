cc_test(test_egr_forward SRCS forward_test.cc DEPS ${eager_deps} ${fluid_deps})
cc_test(test_egr_forward_autograd SRCS forward_autograd_test.cc DEPS ${eager_deps} ${fluid_deps})
