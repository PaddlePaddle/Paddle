function(inference_test TARGET_NAME)
  set(options "")
  set(oneValueArgs "")
  set(multiValueArgs ARGS)
  cmake_parse_arguments(inference_test "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

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
        DEPS paddle_fluid_origin
        ARGS --dirname=${PYTHON_TESTS_DIR}/book/${TARGET_NAME}${arg}.inference.model)
    set_tests_properties(test_inference_${TARGET_NAME}${arg}
            PROPERTIES DEPENDS test_${TARGET_NAME})
    set_tests_properties(test_inference_${TARGET_NAME}${arg}
            PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE")
  endforeach()
endfunction(inference_test)

####################
# Inference tests here depend on fluid/tests/book. If users want to run
# individual test with ctest, they need to run tests in fluid/tests/book
# first to generate saved model.
####################
# This unittest is buggy!
#inference_test(fit_a_line)
inference_test(image_classification ARGS vgg resnet)
inference_test(label_semantic_roles)
inference_test(recognize_digits ARGS mlp conv)
inference_test(recommender_system)
#inference_test(rnn_encoder_decoder)
#inference_test(understand_sentiment ARGS conv)
inference_test(word2vec)

# This is an unly work around to make this test run
# TODO(TJ): clean me up
cc_test(test_inference_nlp
  SRCS test_inference_nlp.cc
  DEPS paddle_fluid_origin
  ARGS
  --model_path=${PADDLE_BINARY_DIR}/python/paddle/fluid/tests/book/recognize_digits_mlp.inference.model)

set_tests_properties(test_train_recognize_digits_conv test_train_recognize_digits_mlp
        PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE")
