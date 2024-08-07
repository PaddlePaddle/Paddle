file(
  GLOB TEST_OPS
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "test_*.py")
string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

foreach(src ${TEST_OPS})
  py_test(${src} SRCS ${src}.py)
endforeach()

set_tests_properties(test_faster_tokenizer_op_deprecated
                     PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE")
set_tests_properties(test_faster_tokenizer_op_deprecated PROPERTIES TIMEOUT 120)
