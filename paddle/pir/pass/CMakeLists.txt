file(GLOB NEW_PASS_SRCS "*.cc")

ir_library(pir_pass SRCS ${NEW_PASS_SRCS} DEPS pir_core)
