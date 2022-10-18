set(CC_TESTS_DIR
    ${PADDLE_BINARY_DIR}/paddle/tests
    CACHE INTERNAL "c++ tests directory")
set(PYTHON_TESTS_DIR
    ${PADDLE_BINARY_DIR}/python/paddle/fluid/tests
    CACHE INTERNAL "python tests directory")

add_subdirectory(utils)
add_subdirectory(scripts)
add_subdirectory(testing)

add_subdirectory(phi)
add_subdirectory(infrt)
add_subdirectory(fluid)

get_property(test_srcs GLOBAL PROPERTY TEST_SRCS)
get_property(test_names GLOBAL PROPERTY TEST_NAMES)
# message("test_srcs ${test_srcs}")

get_property(paddle_lib GLOBAL PROPERTY PADDLE_LIB_NAME)

list(LENGTH test_names len)
if(${len} GREATER_EQUAL 1)
  message("1111")
  math(EXPR stop "${len} - 1")
  foreach(idx RANGE ${stop})
    if(WITH_TESTING)
      list(GET test_srcs ${idx} test_src)
      list(GET test_names ${idx} test_name)
      get_property(test_arg GLOBAL PROPERTY "${test_name}_ARGS")
      message("add test ${test_name}")
      cc_test_build(${test_name} SRCS ${test_src})
      add_dependencies(${test_name} ${paddle_lib})
      if(NOT
         ("${test_names}" STREQUAL "c_broadcast_op_npu_test"
          OR "${test_names}" STREQUAL "c_allreduce_sum_op_npu_test"
          OR "${test_names}" STREQUAL "c_allreduce_max_op_npu_test"
          OR "${test_names}" STREQUAL "c_reducescatter_op_npu_test"
          OR "${test_names}" STREQUAL "c_allgather_op_npu_test"
          OR "${test_names}" STREQUAL "send_v2_op_npu_test"
          OR "${test_names}" STREQUAL "c_reduce_sum_op_npu_test"
          OR "${test_names}" STREQUAL "recv_v2_op_npu_test"))
        cc_test_run(
          ${test_name}
          COMMAND
          ${test_name}
          ARGS
          ${test_arg}
          DIR
          ${CC_TESTS_DIR})
      endif()
    elseif(WITH_TESTING AND NOT TEST ${test_names})
      add_test(NAME ${test_names} COMMAND ${CMAKE_COMMAND} -E echo CI skip
                                          ${test_names}.)
    endif()
    set_target_properties(${test_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                  "${CC_TESTS_DIR}")
  endforeach()
endif()

set_tests_properties(standalone_executor_test PROPERTIES TIMEOUT 100)
add_dependencies(standalone_executor_test download_program)

add_dependencies(layer_test jit_download_program)
add_dependencies(layer_test_new jit_download_program)
set_tests_properties(layer_test_new PROPERTIES ENVIRONMENT
                                               "FLAGS_jit_engine_type=New")

add_dependencies(buddy_allocator_test download_data)

# add target to build all cpp tests
add_custom_target(build_tests)
add_dependencies(build_tests ${test_names})
