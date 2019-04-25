include(operators)
register_operators()

if (WITH_MKLDNN)
    cc_test(test_elementwise_mul_mkldnn_op SRCS mkldnn/elementwise_mul_mkldnn_op_tester.cc DEPS elementwise_mul_op op_test)
endif()
