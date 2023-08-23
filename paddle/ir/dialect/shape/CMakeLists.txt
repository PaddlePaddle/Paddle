file(GLOB_RECURSE SHAPE_SRCS "*.cc")
ir_library(
  ir_shape
  SRCS
  ${SHAPE_SRCS}
  ${PADDLE_SOURCE_DIR}/paddle/fluid/ir/dialect/paddle_dialect/ir/pd_type.cc
  DEPS
  ddim
  ir_core)
