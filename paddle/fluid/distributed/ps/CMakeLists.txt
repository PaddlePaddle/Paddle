set_property(GLOBAL PROPERTY RPC_DEPS sendrecv_rpc ${BRPC_DEPS} string_helper)
add_subdirectory(table)
add_subdirectory(service)
add_subdirectory(wrapper)
