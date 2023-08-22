file(GLOB PD_INTERFACE_SRCS "*.cc")

cc_library(
  pd_trait NOT_FOR_INFER
  SRCS ${PD_INTERFACE_SRCS}
  DEPS ir_core)
