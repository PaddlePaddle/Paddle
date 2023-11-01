file(GLOB PD_INTERFACE_SRCS "*.cc")

cc_library(
  pd_trait
  SRCS ${PD_INTERFACE_SRCS}
  DEPS pir_core)
