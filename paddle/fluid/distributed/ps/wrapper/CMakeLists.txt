
get_property(RPC_DEPS GLOBAL PROPERTY RPC_DEPS)

set_source_files_properties(fleet.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
cc_library(fleet
        SRCS fleet.cc
        DEPS framework_proto ps_framework_proto ps_service variable_helper scope op_registry fs shell ${RPC_DEPS})

target_link_libraries(fleet z)
