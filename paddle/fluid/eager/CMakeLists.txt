add_subdirectory(nodes)
add_subdirectory(api)
add_subdirectory(autocodegen)
add_subdirectory(legacy)


set(eager_deps eager_api utils function_api backward autograd_meta grad_node_info legacy pten)
set(fluid_deps tracer layer proto_desc operator op_registry variable_helper memcpy)
set(generated_deps dygraph_function dygraph_node)

add_subdirectory(generated)
add_subdirectory(tests)

cc_library(autograd_meta SRCS autograd_meta.cc DEPS pten)
cc_library(backward SRCS backward.cc DEPS autograd_meta grad_node_info)
cc_library(gradient_accumulation SRCS gradient_accumulation.cc DEPS blas pten var_type_traits layer math_function)
cc_library(function_api SRCS function_api.cc DEPS pten layer proto_desc operator op_registry variable_helper memcpy scale_op)
cc_library(utils SRCS utils.cc DEPS pten layer proto_desc operator op_registry variable_helper memcpy scale_op)

cc_library(grad_node_info SRCS grad_node_info.cc ${DYGRAPH_BACKWARD_NODES} DEPS function_api pten gradient_accumulation)
cc_library(legacy SRCS ${DYGRAPH_LEGACY} DEPS proto_desc operator pten op_registry variable_helper memcpy)
