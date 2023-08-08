file(GLOB VJP_SRCS "*.cc")

cc_library(
  primitive_vjp_experimental
  SRCS ${VJP_SRCS}
  DEPS primitive_backend_static_experimental)
