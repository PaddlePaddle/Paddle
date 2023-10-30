set(NEWIR_SOURCE_DIR "${PADDLE_SOURCE_DIR}/paddle/pir")
set(NEWIR_BINARY_DIR "${PADDLE_BINARY_DIR}/paddle/pir")

file(GLOB IR_SRCS "*.cc")

file(GLOB IR_PARSER_SRCS "parser/*.cc")
list(APPEND IR_SRCS ${IR_PARSER_SRCS})
set(core_srcs
    ${IR_SRCS}
    CACHE INTERNAL "")
