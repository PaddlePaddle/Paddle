file(GLOB spmd_srcs *.cc)

cc_library(
  spmd_rules
  SRCS ${spmd_srcs}
  DEPS phi)
