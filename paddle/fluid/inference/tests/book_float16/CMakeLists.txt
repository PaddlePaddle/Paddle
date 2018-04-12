function(inference_test TARGET_NAME)
  set(options "")
  set(oneValueArgs "")
  set(multiValueArgs ARGS)
  cmake_parse_arguments(inference_test "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  set(PYTHON_TESTS_DIR ${PADDLE_BINARY_DIR}/python/paddle/fluid/tests)
  set(arg_list "")
  if(inference_test_ARGS)
    foreach(arg ${inference_test_ARGS})
      list(APPEND arg_list "_${arg}")
    endforeach()
  else()
    list(APPEND arg_list "_")
  endif()
  foreach(arg ${arg_list})
    string(REGEX REPLACE "^_$" "" arg "${arg}")
    cc_test(test_inference_${TARGET_NAME}${arg}
        SRCS test_inference_${TARGET_NAME}.cc
        DEPS ARCHIVE_START paddle_fluid ARCHIVE_END
        ARGS --dirname=${PYTHON_TESTS_DIR}/book_float16/${TARGET_NAME}${arg}.inference.model)
    set_tests_properties(test_inference_${TARGET_NAME}${arg}
        PROPERTIES DEPENDS test_${TARGET_NAME})
  endforeach()
endfunction(inference_test)

inference_test(float16_image_classification ARGS vgg resnet)
