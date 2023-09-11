file(GLOB_RECURSE CONTROL_FLOW_SRCS "*.cc")
ir_library(pir_control_flow SRCS ${CONTROL_FLOW_SRCS} DEPS pir_core)
