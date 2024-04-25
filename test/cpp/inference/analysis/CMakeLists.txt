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
