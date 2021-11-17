add_subdirectory(api)

set(eager_deps eager_api utils function_api autograd_meta grad_node_info pten pten_api)
set(fluid_deps tracer layer proto_desc operator op_registry variable_helper memcpy)

add_subdirectory(tests)
