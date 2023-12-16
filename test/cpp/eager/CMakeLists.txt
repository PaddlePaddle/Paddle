set(eager_deps
    phi
    common
    hook_utils
    utils
    global_utils
    backward
    tracer
    layer
    autograd_meta
    eager_nan_inf_utils
    grad_node_info
    grad_tensor_holder
    custom_operator_node)

if(NOT (NOT WITH_PYTHON AND ON_INFER))
  set(eager_deps ${eager_deps} accumulation_node prim_utils)
endif()

set(fluid_deps tracer layer proto_desc operator op_registry variable_helper)
set(generated_deps final_dygraph_function final_dygraph_node dygraph_function
                   dygraph_node)

add_subdirectory(data_structure_tests)
add_subdirectory(task_tests)
add_subdirectory(performance_tests)
