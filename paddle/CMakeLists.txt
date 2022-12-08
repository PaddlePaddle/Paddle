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

# NOTE(zhiqiu): The changes of cc tests
# Before, (1) the source file of cc tests are distributed in different sub-directories,
# (2) the tests are added and configured by calling `cc_test()` in each `CMakeLists.txt`,
# (3) the tests links static libraries of paddle modules,
# (4) the tests binaries are generated in different directories, as the same as the
# folder of source file.

# Now, we want to make all cc tests dynamically linked to the main paddle labrary,
# i.e., `libpaddle.so`, so we changes the logic of (2), (3), (4):
# (2) calling `cc_test()` in each `CMakeLists.txt` will not `exactly` add test, but
# record all tests and its source files, the action of add tests is defered to HERE.
# Why doing so? since the target of `libpaddle.so` is mostly the last target, and
# the tests should be added after that accroding to dependency.
# (3) the tests links dynamic libraries, `libpaddle.so`
# (4) the tests are generated to the same directory, i.e., `CC_TESTS_DIR` defined above.

# Next, (to be discusssed)
# (1) move all source files to same folder,
# (2) naturally, and and configure tests in only one `CMakeLists.txt`,
# (3) cc tests support linking pre-built dynamic libraries. For example, use the dynamic
# library in the installed paddle by `pip`.

# add all tests here
get_property(test_srcs GLOBAL PROPERTY TEST_SRCS)
get_property(test_names GLOBAL PROPERTY TEST_NAMES)
# message("test_srcs ${test_srcs}")

get_property(paddle_lib GLOBAL PROPERTY PADDLE_LIB_NAME)

set(POSTFIX ".so")
if(WIN32)
  set(POSTFIX ".dll")
endif()

list(LENGTH test_names len)
if(${len} GREATER_EQUAL 1)
  message("Total cpp tests using dynamic link: ${len}")
  math(EXPR stop "${len} - 1")
  foreach(idx RANGE ${stop})
    if(WITH_TESTING)
      list(GET test_srcs ${idx} test_src)
      list(GET test_names ${idx} test_name)
      get_property(test_arg GLOBAL PROPERTY "${test_name}_ARGS")
      # message("add test ${test_name}")
      add_executable(${test_name} ${test_src})
      target_link_libraries(${test_name} paddle_gtest_main_new)
      target_link_libraries(${test_name} $<TARGET_LINKER_FILE:${paddle_lib}>)
      add_dependencies(${test_name} ${paddle_lib} paddle_gtest_main_new)
      if(WITH_GPU)
        target_link_libraries(${test_name} ${CUDA_CUDART_LIBRARY}
                              "-Wl,--as-needed")
      endif()
      if(WITH_ROCM)
        target_link_libraries(${test_name} ${ROCM_HIPRTC_LIB})
      endif()
      if(APPLE)
        target_link_libraries(${test_name}
                              "-Wl,-rpath,$<TARGET_FILE_DIR:${paddle_lib}>")
      endif()
      if(NOT ((NOT WITH_PYTHON) AND ON_INFER))
        target_link_libraries(${test_name} ${PYTHON_LIBRARIES})
      endif()
      if(WITH_XPU)
        target_link_libraries(${test_name} xpulib)
      endif()
      if(WITH_MLU)
        target_link_libraries(${test_name} neuware_lib)
      endif()
      if(NOT
         ("${test_name}" STREQUAL "c_broadcast_op_npu_test"
          OR "${test_name}" STREQUAL "c_allreduce_sum_op_npu_test"
          OR "${test_name}" STREQUAL "c_allreduce_max_op_npu_test"
          OR "${test_name}" STREQUAL "c_reducescatter_op_npu_test"
          OR "${test_name}" STREQUAL "c_allgather_op_npu_test"
          OR "${test_name}" STREQUAL "send_v2_op_npu_test"
          OR "${test_name}" STREQUAL "c_reduce_sum_op_npu_test"
          OR "${test_name}" STREQUAL "recv_v2_op_npu_test"))
        cc_test_run(
          ${test_name}
          COMMAND
          ${test_name}
          ARGS
          ${test_arg}
          DIR
          ${CC_TESTS_DIR})
      endif()
    elseif(WITH_TESTING AND NOT TEST ${test_name})
      add_test(NAME ${test_name} COMMAND ${CMAKE_COMMAND} -E echo CI skip
                                         ${test_name}.)
    endif()
    set_target_properties(${test_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                  "${CC_TESTS_DIR}")
  endforeach()
endif()

# set properties for some tests, it should be set after the tests defined.
if(TARGET standalone_executor_test)
  set_tests_properties(standalone_executor_test PROPERTIES TIMEOUT 100)
  if(NOT WIN32)
    add_dependencies(standalone_executor_test download_program)
  endif()
endif()

if(TARGET layer_test)
  add_dependencies(layer_test jit_download_program)
  add_dependencies(layer_test_new jit_download_program)
  set_tests_properties(layer_test_new PROPERTIES ENVIRONMENT
                                                 "FLAGS_jit_engine_type=New")
endif()

if(TEST buddy_allocator_test)
  if(NOT WIN32)
    add_dependencies(buddy_allocator_test download_data)
  endif()
  set_tests_properties(buddy_allocator_test PROPERTIES LABELS
                                                       "RUN_TYPE=EXCLUSIVE")
endif()

add_custom_target(build_tests)
# add target to build all cpp tests
if(${len} GREATER_EQUAL 1)
  add_dependencies(build_tests ${test_names})
endif()
