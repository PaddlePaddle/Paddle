file(
  GLOB TEST_OPS
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "test_*.py")
string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

# swgu98: Temporarily commented on Windows platform
if(WIN32)
  list(REMOVE_ITEM TEST_OPS test_faster_tokenizer_op_deprecated)
  set_tests_properties(test_faster_tokenizer_op_deprecated
                       PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE")
  set_tests_properties(test_faster_tokenizer_op_deprecated PROPERTIES TIMEOUT
                                                                      120)
endif()

foreach(src ${TEST_OPS})
  py_test(${src} SRCS ${src}.py)
endforeach()
