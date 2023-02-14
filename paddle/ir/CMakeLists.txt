if(NOT WITH_NEWIR)
  return()
endif()

message(STATUS, "add subdir type")
add_subdirectory(type)

set(NEWIR_SOURCE_DIR "${PADDLE_SOURCE_DIR}/paddle/ir")
set(NEWIR_BINARY_DIR "${PADDLE_BINARY_DIR}/paddle/ir")

message(STATUS, "cc_library ir_context")
cc_library(
  ir_context
  SRCS ir_context.cc
  DEPS storage_uniquer)

message(STATUS, "cc_test ir_context_test")
cc_test_old(ir_context_test SRCS ir_context_test.cc DEPS ir_context gtest)
