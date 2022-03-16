include(operators)
if(WITH_UNITY_BUILD)
    # Load Unity Build rules for operators in paddle/fluid/operators/amp.
    include(unity_build_rule.cmake)
endif()
op_library(mha_data_prepare_op)
op_library(mha_op)
