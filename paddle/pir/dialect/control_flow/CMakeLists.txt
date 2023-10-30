file(GLOB_RECURSE CONTROL_FLOW_SRCS "*.cc")
set(contro_flow_srcs
    ${CONTROL_FLOW_SRCS}
    CACHE INTERNAL "")
#ir_library(pir_control_flow SRCS ${CONTROL_FLOW_SRCS} DEPS pir_core)
