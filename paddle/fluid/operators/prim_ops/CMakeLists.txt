include(operators)
if(WITH_UNITY_BUILD)
  # Load Unity Build rules for operators in paddle/fluid/operators/prim_ops.
  include(unity_build_rule.cmake)
endif()
register_operators()

set(PRIM_OP_SRCS
    reshape_p_op.cc
    broadcast_p_op.cc
    reduce_p_op.cc
    transpose_p_op.cc
    split_p_op.cc
    concat_p_op.cc
    slice_select_p_op.cc
    slice_assign_p_op.cc
    gather_p_op.cc
    scatter_add_p_op.cc
    add_p_op.cc
    sub_p_op.cc
    mul_p_op.cc
    div_p_op.cc
    sqrt_p_op.cc
    tanh_p_op.cc
    sin_p_op.cc
    cos_p_op.cc
    exp_p_op.cc
    matmul_p_op.cc
    fill_constant_p_op.cc
    log_p_op.cc
    select_p_op.cc
    eq_p_op.cc
    pow_p_op.cc
    max_p_op.cc
    erf_p_op.cc)

cc_test(
  prim_op_test
  SRCS prim_op_test.cc ${PRIM_OP_SRCS}
  DEPS op_registry)
