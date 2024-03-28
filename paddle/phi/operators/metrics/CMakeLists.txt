include(operators)
if(WITH_UNITY_BUILD)
  # Load Unity Build rules for operators in paddle/phi/operators/metrics.
  include(unity_build_rule.cmake)
endif()
register_operators()
