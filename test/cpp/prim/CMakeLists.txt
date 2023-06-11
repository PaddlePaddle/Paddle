set(prim_eager_deps
    phi
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

set(prim_generated_deps final_dygraph_function final_dygraph_node
                        dygraph_function dygraph_node)

cc_test_old(
  test_comp_static
  SRCS
  test_static_prim.cc
  DEPS
  static_utils
  static_prim_api
  generated_op
  prim_utils
  operator
  elementwise_mul_op
  elementwise_sub_op
  elementwise_pow_op
  fill_constant_op
  activation_op
  phi
  static_global_utils
  static_tensor_operants
  generated_static_op)

if(NOT (NOT WITH_PYTHON AND ON_INFER))
  if(WITH_CINN)
    set(prim_eager_deps ${prim_eager_deps} cinn_compiler)
  endif()
  cc_library(
    init_env_utils
    SRCS init_env_utils.cc
    DEPS phi eager_tensor_operants static_tensor_operants)

  cc_test_old(
    test_comp_eager
    SRCS
    test_eager_prim.cc
    DEPS
    fleet_executor
    ${prim_eager_deps}
    ${prim_generated_deps}
    prim_utils
    static_global_utils
    init_env_utils
    python)
endif()
