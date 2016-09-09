add_simple_unittest(test_CommandLineParser)
add_simple_unittest(test_Logging)
add_simple_unittest(test_Thread)
add_simple_unittest(test_StringUtils)
add_simple_unittest(test_CustomStackTrace)

add_executable(
    test_CustomStackTracePrint
    test_CustomStackTracePrint.cpp
)
link_paddle_exe(test_CustomStackTracePrint)
add_test(NAME test_CustomStackTracePrint
    COMMAND ${PROJ_ROOT}/paddle/utils/tests/test_CustomStackTracePrint.sh
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
