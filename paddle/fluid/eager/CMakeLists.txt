set(eager_deps pten pten_api hook_utils tensor_utils utils global_utils backward pten_tensor legacy autograd_meta grad_node_info grad_tensor_holder gradient_accumulation accumulation_node)
set(fluid_deps tracer layer proto_desc operator op_registry variable_helper memcpy)
set(generated_deps dygraph_function dygraph_node)

if(NOT ON_INFER)
    message("Performing Eager Dygraph Auto Code Generation")
    add_subdirectory(auto_code_generator)
endif()

add_subdirectory(api)
add_subdirectory(accumulation)
add_subdirectory(legacy)

cc_library(grad_node_info SRCS grad_node_info.cc DEPS pten pten_api)
cc_library(grad_tensor_holder SRCS grad_tensor_holder.cc DEPS grad_node_info gradient_accumulation)

cc_library(autograd_meta SRCS autograd_meta.cc DEPS pten pten_api)
cc_library(utils SRCS utils.cc DEPS pten pten_api global_utils layer proto_desc operator op_registry variable_helper memcpy scale_op autograd_meta hook_utils)
cc_library(legacy SRCS ${DYGRAPH_LEGACY} DEPS global_utils proto_desc operator pten pten_api op_registry variable_helper memcpy)
cc_library(backward SRCS backward.cc DEPS grad_tensor_holder utils autograd_meta grad_node_info)

add_subdirectory(tests)
