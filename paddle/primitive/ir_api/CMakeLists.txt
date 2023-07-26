file(GLOB IR_API_SRCS "*.cc")

cc_library(
  ir_api
  SRCS ${VJP_SRCS}
  DEPS ir_core pd_dialect)
