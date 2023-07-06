file(GLOB SPMD_SRCS "*.cc")
cc_library(
  spmd_rule
  SRCS ${SPMD_SRCS}
  DEPS phi)
