file(GLOB NEW_PASS_SRCS "*.cc")

cc_library(
  new_pass
  SRCS ${NEW_PASS_SRCS}
  DEPS new_ir)
