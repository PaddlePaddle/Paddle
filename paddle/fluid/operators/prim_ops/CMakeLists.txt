include(operators)
if(WITH_UNITY_BUILD)
  # Load Unity Build rules for operators in paddle/fluid/operators/prim_ops.
  include(unity_build_rule.cmake)
endif()
register_operators()
