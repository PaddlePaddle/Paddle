file(GLOB VJP_SRCS "*.cc")

cc_library(
  primitive_vjp_experimental NOT_FOR_INFER
  SRCS ${VJP_SRCS}
  DEPS primitive_backend_static_experimental)
add_dependencies(primitive_vjp_experimental pd_dialect)
