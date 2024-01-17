if(NOT WITH_ROCM)
  paddle_test(
    if_op_test
    SRCS
    if_op_test.cc
    DEPS
    op_dialect_vjp
    pir_transforms
    op_dialect
    gtest)

  paddle_test(
    while_op_test
    SRCS
    while_op_test.cc
    DEPS
    op_dialect_vjp
    pir_transforms
    op_dialect
    gtest
    device_event_base)
endif()
