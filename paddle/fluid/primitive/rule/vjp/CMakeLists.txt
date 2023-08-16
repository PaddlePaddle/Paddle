file(GLOB VJP_SRCS "vjp.cc")
if(WITH_PYTHON OR NOT ON_INFER)
  cc_library(
    primitive_eager_utils_experimental
    SRCS eager_utils.cc
    DEPS phi common_infer_shape_functions)
endif()
cc_library(
  primitive_static_utils_experimental
  SRCS static_utils.cc
  DEPS phi common_infer_shape_functions)
cc_library(
  primitive_vjp_experimental
  SRCS ${VJP_SRCS}
  DEPS primitive_backend_static_experimental static_global_utils
       primitive_static_utils_experimental)
add_dependencies(primitive_vjp_experimental pd_dialect)
