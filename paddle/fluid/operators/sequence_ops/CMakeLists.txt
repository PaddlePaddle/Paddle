include(operators)
if(WITH_UNITY_BUILD)
  # Load Unity Build rules for operators in paddle/fluid/operators/sequence_ops.
  include(unity_build_rule.cmake)
endif()
register_operators()

if(WITH_UNITY_BUILD)
  target_link_libraries(paddle_operators_sequence_ops_unity phi common)
endif()
