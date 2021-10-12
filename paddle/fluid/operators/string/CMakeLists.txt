include(operators)
register_operators(DEPS op_version_registry utf8proc string_array)
# if(WITH_UNITY_BUILD)
#     target_link_libraries(paddle_operators_string_unity ${OP_HEADER_DEPS} ${COMMON_OP_DEPS})
# endif()
if(WITH_UNITY_BUILD)
    # Load Unity Build rules for operators in paddle/fluid/operators/string.
    include(unity_build_rule.cmake)
endif()
