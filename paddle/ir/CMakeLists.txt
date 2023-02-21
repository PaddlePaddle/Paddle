if(NOT WITH_NEWIR)
  return()
endif()

set(NEWIR_SOURCE_DIR "${PADDLE_SOURCE_DIR}/paddle/ir")
set(NEWIR_BINARY_DIR "${PADDLE_BINARY_DIR}/paddle/ir")

file(GLOB IR_SRCS "*.cc")

file(GLOB IR_TEST_SRCS "*_test.cc")

list(REMOVE_ITEM IR_SRCS ${IR_TEST_SRCS})

cc_library(new_ir SRCS ${IR_SRCS})

cc_test_old(type_test SRCS type_test.cc DEPS new_ir gtest)
