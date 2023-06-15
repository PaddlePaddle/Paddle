file(GLOB NEW_PASS_SRCS "*.cc")

cc_library(
  ir_pass
  SRCS ${NEW_PASS_SRCS}
  DEPS ir_core)
