file(GLOB DRR_SRCS "*.cc" "api/*cc")

ir_library(ir_drr SRCS ${DRR_SRCS} DEPS ir_pattern_rewrite)
