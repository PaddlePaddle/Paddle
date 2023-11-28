set(DECOMP_BASE_SRCS
    ${PADDLE_SOURCE_DIR}/paddle/fluid/primitive/base/decomp_trans.cc)
cc_library(
  decomp_base_utils
  SRCS ${DECOMP_BASE_SRCS}
  DEPS primitive_vjp_experimental)
