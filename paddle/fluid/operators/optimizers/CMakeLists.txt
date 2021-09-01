include(operators)
if(WITH_UNITY_BUILD)
    # Load Unity Build rules for operators in paddle/fluid/operators/optimizers.
    include(unity_build_rule.cmake)
endif()
register_operators()
set_property(TARGET lars_momentum_op APPEND PROPERTY COMPILE_OPTIONS $<$<COMPILE_LANGUAGE:CUDA>:-rdc=true>)
