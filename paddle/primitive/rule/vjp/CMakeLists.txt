file(GLOB VJP_SRCS "*.cc")

cc_library(
  primitive_vjp_experimental
  SRCS ${VJP_SRCS}
  DEPS ir_core phi experimental_static_primitive_backend)
