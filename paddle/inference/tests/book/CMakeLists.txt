function(inference_test TARGET_NAME)
  set(options "")
  set(oneValueArgs "")
  set(multiValueArgs ARGS)
  cmake_parse_arguments(inference_test "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  set(PYTHON_TESTS_DIR ${PADDLE_SOURCE_DIR}/python/paddle/v2/fluid/tests)
  if(inference_test_ARGS)
    foreach(arg ${inference_test_ARGS})
      cc_test(test_inference_${TARGET_NAME}_${arg}
          SRCS test_inference_${TARGET_NAME}.cc
          DEPS ARCHIVE_START paddle_fluid ARCHIVE_END
          ARGS --dirname=${PYTHON_TESTS_DIR}/book/${TARGET_NAME}_${arg}.inference.model)
      set_tests_properties(test_inference_${TARGET_NAME}_${arg}
          PROPERTIES DEPENDS test_${TARGET_NAME})
    endforeach()
  else()
    cc_test(test_inference_${TARGET_NAME}
        SRCS test_inference_${TARGET_NAME}.cc
        DEPS ARCHIVE_START paddle_fluid ARCHIVE_END
        ARGS --dirname=${PYTHON_TESTS_DIR}/book/${TARGET_NAME}.inference.model)
    set_tests_properties(test_inference_${TARGET_NAME}
        PROPERTIES DEPENDS test_${TARGET_NAME})
  endif()
endfunction(inference_test)

inference_test(recognize_digits ARGS mlp)
inference_test(image_classification ARGS vgg resnet)
inference_test(label_semantic_roles)
inference_test(rnn_encoder_decoder)
