cc_test(
  test_elementwise_add_op_inplace
  SRCS test_elementwise_add_op_inplace.cc
  DEPS executor
       op_registry
       elementwise_add_op
       scope
       device_context
       enforce
       common)
cc_test(
  test_elementwise_div_grad_grad
  SRCS test_elementwise_div_grad_grad.cc
  DEPS executor
       op_registry
       elementwise_div_op
       scope
       device_context
       enforce
       common)
cc_test(
  test_elementwise_add_grad_grad
  SRCS test_elementwise_add_grad_grad.cc
  DEPS executor
       op_registry
       elementwise_add_op
       scope
       device_context
       enforce
       common)
