set(PYTHON_TESTS_DIR ${PADDLE_SOURCE_DIR}/python/paddle/v2/fluid/tests)
cc_test(test_inference_recognize_digits_mlp_separate
    SRCS test_inference_recognize_digits.cc
    DEPS ARCHIVE_START paddle_fluid ARCHIVE_END
    ARGS --dirname=${PYTHON_TESTS_DIR}/book/recognize_digits_mlp.inference.model)
cc_test(test_inference_recognize_digits_mlp_combine
    SRCS test_inference_recognize_digits_combinels.cc
    DEPS ARCHIVE_START paddle_fluid ARCHIVE_END
    ARGS --dirname=${PYTHON_TESTS_DIR}/book/recognize_digits_mlp_combine.inference.model --prog_filename=__model__ --param_filename=params.dat)
set_tests_properties(test_inference_recognize_digits_mlp_separate
    PROPERTIES DEPENDS test_recognize_digits)
set_tests_properties(test_inference_recognize_digits_mlp_combine
    PROPERTIES DEPENDS test_recognize_digits)
