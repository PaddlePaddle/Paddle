set(NEWIR_SOURCE_DIR "${PADDLE_SOURCE_DIR}/paddle/ir")
set(NEWIR_BINARY_DIR "${PADDLE_BINARY_DIR}/paddle/ir")

file(GLOB IR_SRCS "*.cc")

file(GLOB IR_PARSER_SRCS "parser/*.cc")

list(APPEND IR_SRCS ${IR_PARSER_SRCS})

ir_library(ir_core SRCS ${IR_SRCS} DEPS ddim)
