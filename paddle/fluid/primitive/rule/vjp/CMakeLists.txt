file(GLOB VJP_SRCS "vjp.cc")
cc_library(
  primitive_vjp_experimental NOT_FOR_INFER
  SRCS ${VJP_SRCS}
  DEPS primitive_backend_static_experimental static_global_utils
       primitive_static_utils_experimental)
