add_simple_unittest(test_Thread)
add_simple_unittest(test_StringUtils)
add_simple_unittest(test_CustomStackTrace)
add_simple_unittest(test_ThreadBarrier)
add_simple_unittest(test_SpinLock)
add_simple_unittest(test_SIMDFlags)

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

add_library(test_class_registrar_lib STATIC
  test_ClassRegistrarLib.cpp
  test_ClassRegistrarGlobals.cpp)

add_simple_unittest(test_ClassRegistrar)
target_link_libraries(test_ClassRegistrar
  test_class_registrar_lib)
