add_executable(test_inference_recognize_digits_mlp test_inference_recognize_digits_mlp.cc)
target_circle_link_libraries(
    test_inference_recognize_digits_mlp
    ARCHIVE_START
    paddle_fluid
    ARCHIVE_END
    gtest
    gflags)
add_test(
    NAME test_inference_recognize_digits_mlp
    COMMAND test_inference_recognize_digits_mlp
           --dirname=${PADDLE_SOURCE_DIR}/python/paddle/v2/fluid/tests/book/recognize_digits_mlp.inference.model
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
