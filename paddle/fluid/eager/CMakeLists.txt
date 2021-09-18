add_subdirectory(nodes)
add_subdirectory(tests)
add_subdirectory(api)
add_subdirectory(autocodegen)

cc_library(autograd_meta SRCS autograd_meta.cc DEPS top)
cc_library(backward SRCS backward.cc DEPS autograd_meta grad_node_info)
cc_library(gradient_accumulation SRCS gradient_accumulation.cc DEPS blas top var_type_traits layer math_function)
cc_library(function_api SRCS function_api.cc DEPS top layer proto_desc operator op_registry variable_helper memcpy scale_op)
cc_library(utils SRCS utils.cc DEPS top layer proto_desc operator op_registry variable_helper memcpy scale_op)

cc_library(grad_node_info SRCS grad_node_info.cc ${DYGRAPH_BACKWARD_NODES} DEPS function_api top gradient_accumulation)
