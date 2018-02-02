set(PYTHON_TESTS_DIR ${PADDLE_SOURCE_DIR}/python/paddle/v2/fluid/tests)
cc_test(test_inference_recognize_digits_mlp
    SRCS test_inference_recognize_digits.cc
    DEPS ARCHIVE_START paddle_fluid ARCHIVE_END
    ARGS --dirname=${PYTHON_TESTS_DIR}/book/recognize_digits_mlp.inference.model)
set_tests_properties(test_inference_recognize_digits_mlp
    PROPERTIES DEPENDS test_recognize_digits)
