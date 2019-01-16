file(GLOB TEST_OPS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "test_*.py")

if(NOT WITH_MKLDNN)
    list(REMOVE_ITEM TEST_OPS "test_calibration.py")
endif()

string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

foreach(src ${TEST_OPS})
    py_test(${src} SRCS ${src}.py)
endforeach()
