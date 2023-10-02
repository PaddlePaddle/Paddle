remove_definitions(-DPADDLE_DLL_EXPORT)
set(CC_TESTS_DIR
    ${PADDLE_BINARY_DIR}/test/cpp
    CACHE INTERNAL "c++ tests directory")
set(PYTHON_TESTS_DIR
    ${PADDLE_BINARY_DIR}/test
    CACHE INTERNAL "python tests directory")

function(py_test_modules TARGET_NAME)
  if(WITH_TESTING)
    set(options SERIAL)
    set(oneValueArgs "")
    set(multiValueArgs MODULES DEPS ENVS)
    cmake_parse_arguments(py_test_modules "${options}" "${oneValueArgs}"
                          "${multiValueArgs}" ${ARGN})

    if(WITH_COVERAGE AND NOT (WITH_INCREMENTAL_COVERAGE
                              AND "$ENV{PADDLE_GIT_DIFF_PY_FILE}" STREQUAL ""))
      add_test(
        NAME ${TARGET_NAME}
        COMMAND
          ${CMAKE_COMMAND} -E env PYTHONPATH=${PADDLE_BINARY_DIR}/python
          ${py_test_modules_ENVS}
          COVERAGE_FILE=${PADDLE_BINARY_DIR}/python-coverage.data
          ${PYTHON_EXECUTABLE} -m coverage run --branch -p
          ${PADDLE_SOURCE_DIR}/tools/test_runner.py ${py_test_modules_MODULES}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    else()
      add_test(
        NAME ${TARGET_NAME}
        COMMAND
          ${CMAKE_COMMAND} -E env PYTHONPATH=${PADDLE_BINARY_DIR}/python
          ${py_test_modules_ENVS} ${PYTHON_EXECUTABLE}
          ${PADDLE_SOURCE_DIR}/tools/test_runner.py ${py_test_modules_MODULES}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    endif()

    if(py_test_modules_SERIAL)
      set_property(TEST ${TARGET_NAME} PROPERTY RUN_SERIAL 1)
    endif()
    if(WIN32)
      set_tests_properties(${TARGET_NAME} PROPERTIES TIMEOUT 150)
    endif()
  endif()
endfunction()

function(bash_test_modules TARGET_NAME)
  if(NOT WITH_TESTING)
    return()
  endif()

  set(options SERIAL)
  set(oneValueArgs TIMEOUT START_BASH)
  set(multiValueArgs DEPS ENVS LABELS)
  cmake_parse_arguments(bash_test_modules "${options}" "${oneValueArgs}"
                        "${multiValueArgs}" ${ARGN})

  set(timeout 350)
  if(${bash_test_modules_TIMEOUT})
    set(timeout ${bash_test_modules_TIMEOUT})
  endif()

  if(WITH_COVERAGE)
    add_test(
      NAME ${TARGET_NAME}
      COMMAND
        ${CMAKE_COMMAND} -E env PYTHONPATH=${PADDLE_BINARY_DIR}/python
        TEST_TARGET_NAME=${TARGET_NAME} TEST_TIMEOUT=${timeout}
        ${bash_test_modules_ENVS} WITH_COVERAGE=ON
        COVERAGE_FILE=${PADDLE_BINARY_DIR}/python-coverage.data bash
        ${CMAKE_CURRENT_BINARY_DIR}/${bash_test_modules_START_BASH}
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  else()
    add_test(
      NAME ${TARGET_NAME}
      COMMAND
        ${CMAKE_COMMAND} -E env PYTHONPATH=${PADDLE_BINARY_DIR}/python
        TEST_TARGET_NAME=${TARGET_NAME} TEST_TIMEOUT=${timeout}
        ${bash_test_modules_ENVS} bash
        ${CMAKE_CURRENT_BINARY_DIR}/${bash_test_modules_START_BASH}
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  endif()

  if(bash_test_modules_SERIAL)
    set_property(TEST ${TARGET_NAME} PROPERTY RUN_SERIAL 1)
  endif()

  if(bash_test_modules_LABELS)
    set_tests_properties(${TARGET_NAME} PROPERTIES LABELS
                                                   ${bash_test_modules_LABELS})
  endif()
endfunction()

if(WITH_TESTING)
  if(WITH_CINN)
    add_subdirectory(cpp/cinn)
    add_subdirectory(cinn)
  endif()
  if(CINN_ONLY)
    return()
  endif()

  add_subdirectory(amp)
  add_subdirectory(asp)
  add_subdirectory(autograd)
  add_subdirectory(book)
  # add_subdirectory(composite_ops)
  add_subdirectory(contrib)
  add_subdirectory(cpp)
  add_subdirectory(custom_kernel)
  add_subdirectory(custom_op)
  add_subdirectory(custom_runtime)
  add_subdirectory(dataset)
  add_subdirectory(cpp_extension)
  if(WITH_DISTRIBUTE)
    add_subdirectory(collective)
    add_subdirectory(auto_parallel)
    add_subdirectory(distributed_passes)
    add_subdirectory(ps)
  endif()
  add_subdirectory(distribution)
  add_subdirectory(dygraph_to_static)
  if(NOT WIN32 OR NOT WITH_GPU)
    add_subdirectory(fft)
  endif()
  # add_subdirectory(fleet)
  if(WITH_IPU)
    add_subdirectory(ipu)
  endif()
  add_subdirectory(ir)
  add_subdirectory(indexing)
  add_subdirectory(legacy_test)
  if(WITH_MKLDNN)
    add_subdirectory(mkldnn)
  endif()
  add_subdirectory(prim)
  add_subdirectory(quantization)
  add_subdirectory(rnn)
  add_subdirectory(rpc)
  add_subdirectory(sequence)
  add_subdirectory(standalone_executor)
  add_subdirectory(tokenizer)
  # add_subdirectory(white_list)
  if(WITH_XPU)
    add_subdirectory(xpu)
  endif()
endif()

if(WITH_CPP_DIST)
  add_test(NAME test_paddle_lib
           COMMAND ${PADDLE_BINARY_DIR}/test/paddle_lib/test_paddle_lib)
  if(WITH_GPU)
    add_test(NAME test_paddle_lib_gpu
             COMMAND ${PADDLE_BINARY_DIR}/test/paddle_lib/test_paddle_lib_gpu)
  endif()
endif()

get_property(test_srcs GLOBAL PROPERTY TEST_SRCS)
get_property(test_names GLOBAL PROPERTY TEST_NAMES)

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
      if(WITH_SHARED_PHI)
        target_link_libraries(${test_name} $<TARGET_LINKER_FILE:phi>)
      endif()
      if(WITH_SHARED_IR)
        target_link_libraries(${test_name} $<TARGET_LINKER_FILE:pir>)
      endif()
      add_dependencies(${test_name} ${paddle_lib} paddle_gtest_main_new)
      if(WITH_GPU)
        target_link_libraries(${test_name} ${CUDA_CUDART_LIBRARY}
                              "-Wl,--as-needed")
      endif()
      if(WITH_ROCM)
        target_link_libraries(${test_name} ${ROCM_HIPRTC_LIB})
      endif()
      if(APPLE)
        target_link_libraries(
          ${test_name}
          "-Wl,-rpath,$<TARGET_FILE_DIR:${paddle_lib}> -Wl,-rpath,$<TARGET_FILE_DIR:phi> -Wl,-rpath,$<TARGET_FILE_DIR:pir>"
        )
      endif()
      if(NOT ((NOT WITH_PYTHON) AND ON_INFER))
        target_link_libraries(${test_name} ${PYTHON_LIBRARIES})
      endif()
      if(WITH_CINN AND NOT CINN_ONLY)
        target_link_libraries(${test_name} $<TARGET_LINKER_FILE:cinnapi>)
      endif()
      if(WITH_XPU)
        target_link_libraries(${test_name} xpulib)
      endif()
      cc_test_run(
        ${test_name}
        COMMAND
        ${test_name}
        ARGS
        ${test_arg}
        DIR
        ${CC_TESTS_DIR})
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
