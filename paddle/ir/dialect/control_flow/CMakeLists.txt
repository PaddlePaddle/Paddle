file(GLOB_RECURSE CONTROL_FLOW_SRCS "*.cc")
ir_library(ir_control_flow SRCS ${CONTROL_FLOW_SRCS} DEPS ir_core)
