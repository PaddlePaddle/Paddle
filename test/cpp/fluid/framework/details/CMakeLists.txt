paddle_test(broadcast_op_test SRCS broadcast_op_handle_test.cc)

paddle_test(gather_op_test SRCS gather_op_handle_test.cc)

paddle_test(
  fused_broadcast_op_test
  SRCS fused_broadcast_op_handle_test.cc
)
paddle_test(exception_holder_test SRCS exception_holder_test.cc)
