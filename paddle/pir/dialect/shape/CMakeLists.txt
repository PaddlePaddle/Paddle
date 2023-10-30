file(GLOB_RECURSE SHAPE_SRCS "*.cc")
set(shape_srcs
    ${SHAPE_SRCS}
    CACHE INTERNAL "")

#ir_library(pir_shape SRCS ${SHAPE_SRCS} DEPS pir_core)
