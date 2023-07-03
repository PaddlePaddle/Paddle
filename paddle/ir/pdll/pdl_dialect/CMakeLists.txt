file(GLOB PATTERN_SRCS "*.cc")

ir_library(pdl SRCS ${PATTERN_SRCS} DEPS ir_core)
