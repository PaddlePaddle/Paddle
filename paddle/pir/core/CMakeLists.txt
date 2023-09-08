set(NEWIR_SOURCE_DIR "${PADDLE_SOURCE_DIR}/paddle/pir")
set(NEWIR_BINARY_DIR "${PADDLE_BINARY_DIR}/paddle/pir")

file(GLOB IR_SRCS "*.cc")

ir_library(pir_core SRCS ${IR_SRCS} DEPS ddim)
