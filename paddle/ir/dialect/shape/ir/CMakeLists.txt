file(GLOB SHAPE_SRCS "*.cc")
ir_library(ir_shape SRCS ${SHAPE_SRCS} DEPS ir_core)
