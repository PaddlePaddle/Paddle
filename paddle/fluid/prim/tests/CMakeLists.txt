set(prim_eager_deps
    phi_api
    phi_dygraph_api
    hook_utils
    tensor_utils
    utils
    global_utils
    backward
    phi_tensor
    tracer
    layer
    autograd_meta
    eager_nan_inf_utils
    grad_node_info
    grad_tensor_holder
    custom_operator_node)

set(prim_generated_deps final_dygraph_function final_dygraph_node
                        dygraph_function dygraph_node)

cc_test_old(
  test_static_prim
  SRCS
  test_static_prim.cc
  DEPS
  static_utils
  static_prim_api
  generated_op
  prim_utils
  operator
  elementwise_mul_op
  scale_op
  activation_op
  phi_api
  phi_dygraph_api
  static_global_utils)

if(NOT (NOT WITH_PYTHON AND ON_INFER))
  cc_test_old(
    test_eager_prim
    SRCS
    test_eager_prim.cc
    DEPS
    ${prim_eager_deps}
    ${prim_generated_deps}
    prim_utils
    static_global_utils)
endif()
