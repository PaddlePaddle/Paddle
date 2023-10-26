file(GLOB PATTERN_SRCS "*.cc")

ir_library(pir_pattern_rewrite SRCS ${PATTERN_SRCS} DEPS pir_core)
