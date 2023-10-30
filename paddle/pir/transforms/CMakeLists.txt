file(GLOB TRANSFORMS_SRCS "*.cc")
set(transforms_srcs
    ${TRANSFORMS_SRCS}
    CACHE INTERNAL "")
# ir_library(
#   pir_builtin_transforms
#   SRCS
#   ${PATTERN_SRCS}
#   DEPS
#   pir_core
#   pir_pattern_rewrite
#   pir_pass)
