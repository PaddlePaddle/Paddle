file(
  GLOB TEST_OPS
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "test_*.py")
string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

# default test
foreach(src ${TEST_OPS})
  py_test(${src} SRCS ${src}.py)
  set_tests_properties(${src} PROPERTIES FIXTURES_SETUP ${src}_infer_model)
endforeach()
set_tests_properties(test_word2vec_book PROPERTIES TIMEOUT 120)
set_tests_properties(test_recognize_digits PROPERTIES TIMEOUT 120)
set_tests_properties(test_image_classification PROPERTIES TIMEOUT 200)
set_tests_properties(test_fit_a_line PROPERTIES TIMEOUT 120)
