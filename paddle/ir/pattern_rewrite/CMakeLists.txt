file(GLOB PATTERN_SRCS "*.cc")

cc_library(
  pattern_rewrite
  SRCS ${PATTERN_SRCS}
  DEPS new_ir)
