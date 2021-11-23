set(eager_deps pten pten_api)

add_subdirectory(data_structure_tests)

cc_test(test_egr_forward_autograd SRCS forward_autograd_test.cc DEPS ${eager_deps} ${fluid_deps})
