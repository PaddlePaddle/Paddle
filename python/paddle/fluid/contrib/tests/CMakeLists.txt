file(GLOB TEST_OPS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "test_*.py")
string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

if(NOT WITH_MKLDNN)
    list(REMOVE_ITEM TEST_OPS test_calibration)
endif()

foreach(src ${TEST_OPS})
    py_test(${src} SRCS ${src}.py)
endforeach()
