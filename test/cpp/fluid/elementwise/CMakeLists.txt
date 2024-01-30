paddle_test(
  test_elementwise_add_op_inplace
  SRCS test_elementwise_add_op_inplace.cc
  DEPS common)
paddle_test(
  test_elementwise_div_grad_grad
  SRCS test_elementwise_div_grad_grad.cc
  DEPS common)
paddle_test(
  test_elementwise_add_grad_grad
  SRCS test_elementwise_add_grad_grad.cc
  DEPS common)
