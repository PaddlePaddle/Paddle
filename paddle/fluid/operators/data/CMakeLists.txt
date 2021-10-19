include(operators)
if(WITH_UNITY_BUILD)
    # Load Unity Build rules for operators in paddle/fluid/operators/data_ops.
    include(unity_build_rule.cmake)
endif()

register_operators()

# TODO: add test here
# cc_test(xxx SRCS xxx DEPS xxx)
