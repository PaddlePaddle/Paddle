file(GLOB TEST_OPS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "test_*.py")
string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

foreach(TEST_OP ${TEST_OPS})
    py_test_modules(${TEST_OP} MODULES ${TEST_OP})
endforeach(TEST_OP)

set_tests_properties(test_se_resnet PROPERTIES TIMEOUT 900)
set_tests_properties(test_tsm PROPERTIES TIMEOUT 900)
set_tests_properties(test_yolov3 PROPERTIES TIMEOUT 900)
