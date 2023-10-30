file(GLOB NEW_PASS_SRCS "*.cc")
set(pass_srcs
    ${NEW_PASS_SRCS}
    CACHE INTERNAL "")
#ir_library(pir_pass SRCS ${NEW_PASS_SRCS} DEPS pir_core)
