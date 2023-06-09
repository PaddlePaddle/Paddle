set(NEWIR_SOURCE_DIR "${PADDLE_SOURCE_DIR}/paddle/ir")
set(NEWIR_BINARY_DIR "${PADDLE_BINARY_DIR}/paddle/ir")

file(GLOB IR_SRCS "*.cc")

cc_library(ir_shared SHARED SRCS ${IR_SRCS})
