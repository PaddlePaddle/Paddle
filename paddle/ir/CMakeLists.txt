if(NOT WITH_NEWIR)
  return()
endif()

add_subdirectory(type)

set(NEWIR_SOURCE_DIR "${PADDLE_SOURCE_DIR}/paddle/ir")
set(NEWIR_BINARY_DIR "${PADDLE_BINARY_DIR}/paddle/ir")

cc_library(
  ir_context
  SRCS ir_context.cc
  DEPS storage_uniquer)

cc_test_old(ir_context_test SRCS ir_context_test.cc DEPS ir_context gtest)
