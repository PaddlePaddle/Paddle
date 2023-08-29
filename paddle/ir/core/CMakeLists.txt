set(NEWIR_SOURCE_DIR "${PADDLE_SOURCE_DIR}/paddle/ir")
set(NEWIR_BINARY_DIR "${PADDLE_BINARY_DIR}/paddle/ir")

file(GLOB IR_SRCS "*.cc")

ir_library(ir_core SRCS ${IR_SRCS} DEPS ddim)
