include(operators)
if(WITH_UNITY_BUILD)
    # Load Unity Build rules for operators in paddle/fluid/operators/amp.
    include(unity_build_rule.cmake)
endif()
register_operators()
