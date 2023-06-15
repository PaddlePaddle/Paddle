file(GLOB PATTERN_SRCS "*.cc")

ir_library(ir_pattern_rewrite SRCS ${PATTERN_SRCS} DEPS ir_core)
