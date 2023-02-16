if(NOT WITH_NEWIR)
  return()
endif()

set(NEWIR_SOURCE_DIR "${PADDLE_SOURCE_DIR}/paddle/ir")
set(NEWIR_BINARY_DIR "${PADDLE_BINARY_DIR}/paddle/ir")

cc_library(storage_uniquer SRCS storage_uniquer.cc)

cc_library(
  ir_context
  SRCS ir_context.cc
  DEPS storage_uniquer)

cc_test_old(type_test SRCS type_test.cc DEPS ir_context gtest)
