set(PYTHON_TESTS_DIR ${PADDLE_SOURCE_DIR}/python/paddle/v2/fluid/tests)
add_executable(test_inference_recognize_digits test_inference_recognize_digits.cc)
target_circle_link_libraries(
    test_inference_recognize_digits
    ARCHIVE_START
    paddle_fluid
    ARCHIVE_END
    gtest
    gflags)
add_test(
    NAME test_inference_recognize_digits_mlp
    COMMAND test_inference_recognize_digits
            --dirname=${PYTHON_TESTS_DIR}/book/recognize_digits_mlp.inference.model
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
set_tests_properties(test_inference_recognize_digits_mlp
    PROPERTIES DEPENDS test_recognize_digits_mlp_cpu)
