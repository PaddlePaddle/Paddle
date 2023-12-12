if(NOT WITH_ROCM)
  cc_test_old(
    if_op_test
    SRCS
    if_op_test.cc
    DEPS
    pir
    op_dialect_vjp
    pir_transforms
    op_dialect
    gtest)

  cc_test_old(
    while_op_test
    SRCS
    while_op_test.cc
    DEPS
    pir
    op_dialect_vjp
    pir_transforms
    op_dialect
    gtest)
endif()
