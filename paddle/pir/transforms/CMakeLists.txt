file(GLOB PATTERN_SRCS "*.cc")

ir_library(
  pir_builtin_transforms
  SRCS
  ${PATTERN_SRCS}
  DEPS
  pir_core
  pir_pattern_rewrite
  pir_pass)
