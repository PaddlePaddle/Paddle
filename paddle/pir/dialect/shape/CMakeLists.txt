file(GLOB_RECURSE SHAPE_SRCS "*.cc")
ir_library(pir_shape SRCS ${SHAPE_SRCS} DEPS pir_core)
