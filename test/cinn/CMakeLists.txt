set(CINN_PYTHON_TEST_DIR ${CMAKE_BINARY_DIR}/test/cinn)

add_custom_command(
  OUTPUT ${CMAKE_BINARY_DIR}/test/__init__.py POST_BUILD
  COMMAND cp -rf --remove-destination ${PROJECT_SOURCE_DIR}/test/cinn
          ${CMAKE_BINARY_DIR}/test/
  COMMAND cd ${CMAKE_BINARY_DIR}/test/ && touch __init__.py)
add_custom_target(COPY_CINN_PYTHON_TESTS ALL
                  DEPENDS ${CMAKE_BINARY_DIR}/test/__init__.py)

set(BASIC_TEST_NAMES test_common test_packed_func test_op_broadcast)

foreach(basic_test_name ${BASIC_TEST_NAMES})
  add_test(
    NAME ${basic_test_name}
    COMMAND
      ${CMAKE_COMMAND} -E env
      PYTHONPATH=${CMAKE_BINARY_DIR}:${CMAKE_BINARY_DIR}/python:$ENV{PYTHONPATH}
      python3 ${CMAKE_CURRENT_SOURCE_DIR}/${basic_test_name}.py
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
endforeach()

add_test(
  NAME test_cinn_op_benchmark
  COMMAND
    ${CMAKE_COMMAND} -E env
    PYTHONPATH=${CMAKE_BINARY_DIR}:${CMAKE_BINARY_DIR}/python:$ENV{PYTHONPATH}
    python3 ${CMAKE_CURRENT_SOURCE_DIR}/test_op_benchmark.py "${WITH_GPU}"
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

if(WITH_CUDNN)
  add_test(
    NAME test_cinn_fake_resnet
    COMMAND
      ${CMAKE_COMMAND} -E env
      PYTHONPATH=${CMAKE_BINARY_DIR}:${CMAKE_BINARY_DIR}/python:$ENV{PYTHONPATH}
      python3 ${CMAKE_CURRENT_SOURCE_DIR}/test_resnet.py
      "${CMAKE_BINARY_DIR}/third_party/resnet_model" "${WITH_GPU}"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

  add_test(
    NAME test_paddle_model_convertor
    COMMAND
      ${CMAKE_COMMAND} -E env
      PYTHONPATH=${CMAKE_BINARY_DIR}:${CMAKE_BINARY_DIR}/python:$ENV{PYTHONPATH}
      FLAGS_cinn_infer_model_version=1.0 python3
      ${CMAKE_CURRENT_SOURCE_DIR}/test_paddle_model_convertor.py --path
      "${CMAKE_BINARY_DIR}/third_party/resnet_model_1"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
endif()

if(WITH_GPU)
  add_test(
    NAME test_launch_kernel_with_symbol
    COMMAND
      ${CMAKE_COMMAND} -E env
      PYTHONPATH=${CMAKE_BINARY_DIR}:${CMAKE_BINARY_DIR}/python/:$ENV{PYTHONPATH}
      python3 ${CMAKE_CURRENT_SOURCE_DIR}/test_launch_kernel_with_symbol.py
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

  file(
    GLOB CINN_IR_TEST
    RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "ir/test_*.py")

  foreach(ir_test_name ${CINN_IR_TEST})
    string(REGEX REPLACE ".py" "" ir_test_name ${ir_test_name})
    add_test(
      NAME ${ir_test_name}
      COMMAND
        ${CMAKE_COMMAND} -E env
        PYTHONPATH=${CMAKE_BINARY_DIR}:${CMAKE_BINARY_DIR}/python/:$ENV{PYTHONPATH}
        python3 ${CMAKE_CURRENT_SOURCE_DIR}/${ir_test_name}.py
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
  endforeach()

endif()
