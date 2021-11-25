set(eager_deps pten pten_api utils global_utils pten_tensor autograd_meta grad_node_info grad_tensor_holder gradient_accumulation accumulation_node)
set(fluid_deps tracer layer proto_desc operator op_registry variable_helper memcpy)

add_subdirectory(data_structure_tests)
add_subdirectory(task_tests)

cc_test(test_egr_forward_autograd SRCS forward_autograd_test.cc DEPS ${eager_deps} ${fluid_deps})
