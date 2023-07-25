file(GLOB VJP_SRCS "*.cc")

cc_library(
  vjp
  SRCS ${VJP_SRCS}
  DEPS ir_core)
