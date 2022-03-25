cc_test(ivalue_test SRCS ivalue_test.cc DEPS phi phi_api_utils phi_tensor_raw)
set(JIT_DEPS executor scope proto_desc phi phi_api_utils phi_tensor_raw
    elementwise_add_op matmul_v2_op activation_op reduce_mean_op feed_op fetch_op)
cc_test(layer_test SRCS layer_test.cc DEPS ${JIT_DEPS})
