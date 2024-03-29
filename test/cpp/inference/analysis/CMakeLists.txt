function(inference_analysis_test_build TARGET)
  if(WITH_TESTING)
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs SRCS EXTRA_DEPS)
    cmake_parse_arguments(analysis_test "${options}" "${oneValueArgs}"
                          "${multiValueArgs}" ${ARGN})
    inference_base_test_build(${TARGET} SRCS ${analysis_test_SRCS} DEPS
                              ${analysis_test_EXTRA_DEPS})
  endif()
endfunction()

function(inference_analysis_test_run TARGET)
  if(WITH_TESTING)
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs COMMAND ARGS)
    cmake_parse_arguments(analysis_test "${options}" "${oneValueArgs}"
                          "${multiValueArgs}" ${ARGN})
    inference_base_test_run(${TARGET} COMMAND ${analysis_test_COMMAND} ARGS
                            ${analysis_test_ARGS})
    set_tests_properties(${TARGET} PROPERTIES LABELS "RUN_TYPE=INFER")
  endif()
endfunction()

function(inference_analysis_test TARGET)
  if(WITH_TESTING)
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs SRCS ARGS EXTRA_DEPS)
    cmake_parse_arguments(analysis_test "${options}" "${oneValueArgs}"
                          "${multiValueArgs}" ${ARGN})
    inference_base_test_build(${TARGET} SRCS ${analysis_test_SRCS} DEPS
                              ${analysis_test_EXTRA_DEPS})
    inference_base_test_run(${TARGET} COMMAND ${TARGET} ARGS
                            ${analysis_test_ARGS})
    set_tests_properties(${TARGET} PROPERTIES LABELS "RUN_TYPE=INFER")
  endif()
endfunction()

if(NOT APPLE)
  inference_analysis_test(
    test_analyzer
    SRCS
    analyzer_tester.cc
    EXTRA_DEPS
    common
    paddle_inference_shared
    ARGS
    --inference_model_dir=${WORD2VEC_MODEL_DIR})
  if(WITH_ONNXRUNTIME AND WIN32)
    # Copy onnxruntime for some c++ test in Windows, since the test will
    # be build only in CI, so suppose the generator in Windows is Ninja.
    copy_onnx(test_analyzer)
  endif()
endif()
