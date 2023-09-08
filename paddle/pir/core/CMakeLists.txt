set(NEWIR_SOURCE_DIR "${PADDLE_SOURCE_DIR}/paddle/pir")
set(NEWIR_BINARY_DIR "${PADDLE_BINARY_DIR}/paddle/pir")

file(GLOB IR_SRCS "*.cc")

file(GLOB IR_PARSER_SRCS "parser/*.cc")
list(APPEND IR_SRCS ${IR_PARSER_SRCS})

ir_library(pir_core SRCS ${IR_SRCS} DEPS ddim)
