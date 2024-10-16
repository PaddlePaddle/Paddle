include(operators)
if(WITH_UNITY_BUILD)
  # Load Unity Build rules for operators in paddle/fluid/operators/sequence_ops.
  include(unity_build_rule.cmake)
endif()
register_operators(DEPS op_version_registry phi)
