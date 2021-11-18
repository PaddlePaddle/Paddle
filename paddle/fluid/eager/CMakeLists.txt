add_subdirectory(api)

set(eager_deps eager_api utils function_api autograd_meta grad_node_info pten pten_api)
set(fluid_deps tracer layer proto_desc operator op_registry variable_helper memcpy)

cc_library(autograd_meta SRCS autograd_meta.cc DEPS pten pten_api)
cc_library(grad_node_info SRCS grad_node_info.cc DEPS function_api pten pten_api)
cc_library(utils SRCS utils.cc DEPS pten pten_api layer proto_desc operator op_registry variable_helper memcpy scale_op)
cc_library(gradient_accumulation SRCS gradient_accumulation.cc DEPS blas pten pten_api var_type_traits layer math_function)

add_subdirectory(tests)
