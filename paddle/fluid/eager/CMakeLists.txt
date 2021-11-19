set(eager_deps eager_api backward utils autograd_meta grad_node_info legacy pten pten_api)
set(fluid_deps tracer layer proto_desc operator op_registry variable_helper memcpy)
set(generated_deps dygraph_function dygraph_node)

add_subdirectory(auto_code_generator)
add_subdirectory(api)
add_subdirectory(accumulation)
add_subdirectory(legacy)

cc_library(grad_node_info SRCS grad_node_info.cc DEPS gradient_accumulation pten pten_api)

cc_library(autograd_meta SRCS autograd_meta.cc DEPS pten pten_api)
cc_library(utils SRCS utils.cc DEPS pten pten_api layer proto_desc operator op_registry variable_helper memcpy scale_op accumulation_node autograd_meta)
cc_library(legacy SRCS ${DYGRAPH_LEGACY} DEPS global_utils proto_desc operator pten pten_api op_registry variable_helper memcpy) 
cc_library(backward SRCS backward.cc DEPS utils autograd_meta grad_node_info)

add_subdirectory(tests)
