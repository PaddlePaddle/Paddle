set(eager_deps phi_api phi_dygraph_api hook_utils tensor_utils utils global_utils backward phi_tensor tracer layer autograd_meta grad_node_info grad_tensor_holder accumulation_node custom_operator_node)

set(fluid_deps tracer layer proto_desc operator op_registry variable_helper memcpy)
set(generated_deps final_dygraph_function final_dygraph_node dygraph_function dygraph_node)

if(NOT ((NOT WITH_PYTHON) AND ON_INFER))
    message("Performing Eager Dygraph Auto Code Generation")
    add_subdirectory(auto_code_generator)
endif()

add_subdirectory(api)
add_subdirectory(accumulation)
add_subdirectory(custom_operator)


cc_library(grad_node_info SRCS grad_node_info.cc DEPS phi_api phi_tensor)
cc_library(grad_tensor_holder SRCS grad_tensor_holder.cc DEPS grad_node_info gradient_accumulator)

cc_library(autograd_meta SRCS autograd_meta.cc DEPS phi_api phi_tensor)
cc_library(utils SRCS utils.cc DEPS phi_api phi_tensor global_utils layer proto_desc operator op_registry variable_helper memcpy scale_op autograd_meta hook_utils)
cc_library(backward SRCS backward.cc DEPS grad_tensor_holder utils autograd_meta grad_node_info)

add_subdirectory(tests)
