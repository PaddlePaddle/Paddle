file(GLOB PATTERN_SRCS "*.cc")
set(pattern_rewrite_srcs
    ${PATTERN_SRCS}
    CACHE INTERNAL "")
#ir_library(pir_pattern_rewrite SRCS ${PATTERN_SRCS} DEPS pir_core)
