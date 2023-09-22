set(VJP_SRCS
    ${PADDLE_SOURCE_DIR}/paddle/fluid/primitive/rule/vjp/generated/generated_vjp.cc
    ${PADDLE_SOURCE_DIR}/paddle/fluid/primitive/rule/vjp/manual/manual_vjp.cc)
cc_library(
  primitive_vjp_experimental
  SRCS ${VJP_SRCS}
  DEPS primitive_backend_static_experimental static_global_utils
       primitive_static_utils_experimental pd_op_dialect_core)
