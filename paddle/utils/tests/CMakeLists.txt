add_simple_unittest(test_Thread)
add_simple_unittest(test_StringUtils)
add_simple_unittest(test_CustomStackTrace)
add_simple_unittest(test_ThreadBarrier)
add_simple_unittest(test_SpinLock)
add_simple_unittest(test_SIMDFlags)
add_simple_unittest(test_Error)

add_executable(
    test_CustomStackTracePrint
    test_CustomStackTracePrint.cpp
)
link_paddle_exe(test_CustomStackTracePrint)
if(NOT APPLE)
    add_test(NAME test_CustomStackTracePrint
        COMMAND ${PROJ_ROOT}/paddle/utils/tests/test_CustomStackTracePrint.sh
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endif()
