

get_property(RPC_DEPS GLOBAL PROPERTY RPC_DEPS)

set_source_files_properties(communicator.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})


cc_library(communicator SRCS communicator.cc DEPS scope client boost table math_function selected_rows_functor ${RPC_DEPS})
