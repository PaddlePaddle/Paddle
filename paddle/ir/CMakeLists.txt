if(NOT WITH_NEWIR)
  return()
endif()

set(NEWIR_SOURCE_DIR "${PADDLE_SOURCE_DIR}/paddle/ir")
set(NEWIR_BINARY_DIR "${PADDLE_BINARY_DIR}/paddle/ir")

file(GLOB IR_SRCS "*.cc")

cc_library(new_ir SRCS ${IR_SRCS})
