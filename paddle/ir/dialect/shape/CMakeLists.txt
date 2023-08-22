file(GLOB_RECURSE SHAPE_SRCS "*.cc")
ir_library(ir_shape SRCS ${SHAPE_SRCS} DEPS pd_type ir_core)
