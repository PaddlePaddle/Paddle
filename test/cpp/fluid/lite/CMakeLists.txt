get_property(paddle_lib GLOBAL PROPERTY PADDLE_LIB_NAME)

cc_test_old(test_lite_engine_op SRCS lite_engine_op_test.cc DEPS ${paddle_lib}
            python)
