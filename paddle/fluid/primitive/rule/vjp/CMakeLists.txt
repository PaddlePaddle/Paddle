file(GLOB VJP_SRCS "*.cc")

cc_library(
  primitive_vjp_experimental
  SRCS ${VJP_SRCS}
  DEPS ir_core phi primitive_backend_static_experimental type_info static_utils)
