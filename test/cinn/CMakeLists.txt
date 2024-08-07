set(CINN_PYTHON_TEST_DIR ${CMAKE_BINARY_DIR}/test/cinn)

add_custom_command(
  OUTPUT ${CMAKE_BINARY_DIR}/test/__init__.py POST_BUILD
  COMMAND cp -rf --remove-destination ${PROJECT_SOURCE_DIR}/test/cinn
          ${CMAKE_BINARY_DIR}/test/
  COMMAND cd ${CMAKE_BINARY_DIR}/test/ && touch __init__.py)
add_custom_target(COPY_CINN_PYTHON_TESTS ALL
                  DEPENDS ${CMAKE_BINARY_DIR}/test/__init__.py)

set(BASIC_TEST_NAMES test_common test_packed_func
                     #    test_op_transform
)

foreach(basic_test_name ${BASIC_TEST_NAMES})
  add_test(
    NAME ${basic_test_name}
    COMMAND
      ${CMAKE_COMMAND} -E env
      PYTHONPATH=${CMAKE_BINARY_DIR}:${CMAKE_BINARY_DIR}/python:$ENV{PYTHONPATH}
      python3 ${CMAKE_CURRENT_SOURCE_DIR}/${basic_test_name}.py
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
endforeach()

if(NOT ${WITH_GPU})
  #    ADD_TEST(NAME test_op_nn
  #        COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_BINARY_DIR}:${CMAKE_BINARY_DIR}/python/cinn:$ENV{PYTHONPATH}
  #        python3 ${CMAKE_CURRENT_SOURCE_DIR}/test_op_nn.py WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  #    )
endif()

#ADD_TEST(NAME test_computation_python
#    COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_BINARY_DIR}:${CMAKE_BINARY_DIR}/python/cinn:$ENV{PYTHONPATH}
#    python3 ${CMAKE_CURRENT_SOURCE_DIR}/test_computation.py
#    ${CMAKE_BINARY_DIR}/third_party/naive_mul_model
#    "${WITH_GPU}" WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
#)

#ADD_TEST(NAME test_cinn_ops_check
#    COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_BINARY_DIR}:${CMAKE_BINARY_DIR}/python/cinn:$ENV{PYTHONPATH}
#    python3 ${CMAKE_CURRENT_SOURCE_DIR}/test_ops.py "${WITH_GPU}"
#    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
#)

#ADD_TEST(NAME test_cinn_real_facedet
#    COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_BINARY_DIR}:${CMAKE_BINARY_DIR}/python/cinn:$ENV{PYTHONPATH}
#    python3 ${CMAKE_CURRENT_SOURCE_DIR}/test_facedet.py "${CMAKE_BINARY_DIR}/third_party/FaceDet" "${WITH_GPU}"
#    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
if(WITH_GPU)
  file(
    GLOB CINN_OP_TEST
    RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "ops/test_*.py")
  set(EXCLUDE_OP test_conv2d_op)

  foreach(op_test_name ${EXCLUDE_OP})
    list(REMOVE_ITEM CINN_OP_TEST ops/${op_test_name}.py)
  endforeach()

  foreach(op_test_name ${CINN_OP_TEST})
    string(REGEX REPLACE ".py" "" op_test_name ${op_test_name})
    add_test(
      NAME ${op_test_name}
      COMMAND
        ${CMAKE_COMMAND} -E env
        PYTHONPATH=${CMAKE_BINARY_DIR}:${CMAKE_BINARY_DIR}/python:$ENV{PYTHONPATH}
        python3 ${CMAKE_CURRENT_SOURCE_DIR}/${op_test_name}.py
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
  endforeach()

  # test op mapper
  file(
    GLOB CINN_OP_MAPPER_TEST
    RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "op_mappers/test_*.py")
  set(EXCLUDE_OP_MAPPER test_mul_op test_conv2d_op)

  foreach(op_mapper_test_name ${EXCLUDE_OP_MAPPER})
    list(REMOVE_ITEM CINN_OP_MAPPER_TEST op_mappers/${op_mapper_test_name}.py)
  endforeach()

  foreach(op_mapper_test_name ${CINN_OP_MAPPER_TEST})
    string(REGEX REPLACE ".py" "" op_mapper_test_name ${op_mapper_test_name})
    add_test(
      NAME "${op_mapper_test_name}_mapper"
      COMMAND
        ${CMAKE_COMMAND} -E env
        PYTHONPATH=${CMAKE_BINARY_DIR}:${CMAKE_BINARY_DIR}/python:$ENV{PYTHONPATH}
        python3 ${CMAKE_CURRENT_SOURCE_DIR}/${op_mapper_test_name}.py
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
  endforeach()

  # test pass test
  file(
    GLOB CINN_PASS_TEST
    RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "passes/test_*.py")

  set(EXCLUDE_PASS ${EXCLUDE_PASS} test_fusion_fallback_pass)
  set(EXCLUDE_PASS ${EXCLUDE_PASS} test_accuracy_check_pass)

  foreach(pass_test_name ${EXCLUDE_PASS})
    list(REMOVE_ITEM CINN_PASS_TEST passes/${pass_test_name}.py)
  endforeach()

  foreach(pass_test_name ${CINN_PASS_TEST})
    string(REGEX REPLACE ".py" "" pass_test_name ${pass_test_name})
    add_test(
      NAME ${pass_test_name}
      COMMAND
        ${CMAKE_COMMAND} -E env
        PYTHONPATH=${CMAKE_BINARY_DIR}:${CMAKE_BINARY_DIR}/python:$ENV{PYTHONPATH}
        python3 ${CMAKE_CURRENT_SOURCE_DIR}/${pass_test_name}.py
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
  endforeach()

  # test fusion_fallback_pass
  add_test(
    NAME test_fusion_fallback_pass
    COMMAND
      ${CMAKE_COMMAND} -E env
      PYTHONPATH=${CMAKE_BINARY_DIR}:${CMAKE_BINARY_DIR}/python/:$ENV{PYTHONPATH}
      FLAGS_prim_enable_dynamic=1 FLAGS_cinn_new_group_scheduler=1
      FLAGS_check_infer_symbolic=1 FLAGS_enable_pir_api=1
      FLAGS_cinn_bucket_compile=1 FLAGS_group_schedule_tiling_first=1
      FLAGS_cudnn_deterministic=true FLAGS_enable_fusion_fallback=1
      ${PYTHON_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/passes/test_fusion_fallback_pass.py
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

  # test accuracy_check_pass
  add_test(
    NAME test_accuracy_check_pass
    COMMAND
      ${CMAKE_COMMAND} -E env
      PYTHONPATH=${CMAKE_BINARY_DIR}:${CMAKE_BINARY_DIR}/python/:$ENV{PYTHONPATH}
      FLAGS_prim_enable_dynamic=1 FLAGS_cinn_new_group_scheduler=1
      FLAGS_check_infer_symbolic=1 FLAGS_enable_pir_api=1
      FLAGS_cinn_bucket_compile=1 FLAGS_group_schedule_tiling_first=1
      FLAGS_cudnn_deterministic=true FLAGS_enable_cinn_accuracy_check=1
      ${PYTHON_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/passes/test_accuracy_check_pass.py
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

  file(
    GLOB CINN_RUNTIME_TEST
    RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "runtime/test_*.py")

  foreach(runtime_test_name ${EXCLUDE_RUNTIME})
    list(REMOVE_ITEM CINN_RUNTIME_TEST runtime/${runtime_test_name}.py)
  endforeach()

  foreach(runtime_test_name ${CINN_RUNTIME_TEST})
    string(REGEX REPLACE ".py" "" runtime_test_name ${runtime_test_name})
    add_test(
      NAME ${runtime_test_name}
      COMMAND
        ${CMAKE_COMMAND} -E env
        PYTHONPATH=${CMAKE_BINARY_DIR}:${CMAKE_BINARY_DIR}/python/:$ENV{PYTHONPATH}
        python3 ${CMAKE_CURRENT_SOURCE_DIR}/${runtime_test_name}.py
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
  endforeach()

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
