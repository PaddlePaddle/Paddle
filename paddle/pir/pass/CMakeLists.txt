file(GLOB NEW_PASS_SRCS "*.cc")

ir_library(ir_pass SRCS ${NEW_PASS_SRCS} DEPS ir_core)
