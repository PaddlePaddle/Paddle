if(NOT WITH_ROCM)
  paddle_test(
    if_op_test
    SRCS
    if_op_test.cc
    DEPS
    op_dialect_vjp)

  paddle_test(
    while_op_test
    SRCS
    while_op_test.cc
    DEPS
    op_dialect_vjp
    device_event_base)
endif()
