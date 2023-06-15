file(GLOB PATTERN_SRCS "*.cc")

cc_library(
  ir_pattern_rewrite
  SRCS ${PATTERN_SRCS}
  DEPS ir_core)
