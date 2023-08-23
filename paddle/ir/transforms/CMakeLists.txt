file(GLOB PATTERN_SRCS "*.cc")

ir_library(
  ir_builtin_transforms
  SRCS
  ${PATTERN_SRCS}
  DEPS
  ir_core
  ir_pattern_rewrite
  ir_pass)
