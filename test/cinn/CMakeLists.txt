set(CINN_PYTHON_TEST_DIR ${CMAKE_BINARY_DIR}/test/cinn)
set(CINN_CORE_API ${CMAKE_BINARY_DIR}/python/core_api.so)

add_custom_command(
  OUTPUT ${CMAKE_BINARY_DIR}/test/__init__.py POST_BUILD
  COMMAND cp -rf --remove-destination ${PROJECT_SOURCE_DIR}/test/cinn
          ${CMAKE_BINARY_DIR}/test/
  COMMAND cd ${CMAKE_BINARY_DIR}/test/ && touch __init__.py)
add_custom_target(COPY_CINN_PYTHON_TESTS ALL
                  DEPENDS ${CMAKE_BINARY_DIR}/test/__init__.py)

set(BASIC_TEST_NAMES
    test_matmul
    test_common
    test_packed_func
    test_pe_elementwise
    test_pe_reduction
    test_pe_transform
    test_op_broadcast
    #    test_op_transform
)

foreach(basic_test_name ${BASIC_TEST_NAMES})
  add_test(
    NAME ${basic_test_name}
    COMMAND
      ${CMAKE_COMMAND} -E env
      PYTHONPATH=${CMAKE_BINARY_DIR}:${CMAKE_BINARY_DIR}/python/cinn:$ENV{PYTHONPATH}
      python3 ${CMAKE_CURRENT_SOURCE_DIR}/${basic_test_name}.py
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
endforeach()

if(NOT ${WITH_GPU})
  #    ADD_TEST(NAME test_op_nn
  #        COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_BINARY_DIR}:${CMAKE_BINARY_DIR}/python/cinn:$ENV{PYTHONPATH}
  #        python3 ${CMAKE_CURRENT_SOURCE_DIR}/test_op_nn.py WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  #    )
endif()

if(WITH_CUDNN)
  # TODO(thisjiang): revert test_cinn_frontend after fix inference mul problem
  # ADD_TEST(NAME test_cinn_frontend
  #     COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_BINARY_DIR}:${CMAKE_BINARY_DIR}/python/cinn:$ENV{PYTHONPATH}
  #     python3 ${CMAKE_CURRENT_SOURCE_DIR}/test_frontend.py
  #     ${CMAKE_BINARY_DIR}/third_party/naive_mul_model
  #     ${CMAKE_BINARY_DIR}/third_party/multi_fc_model
  #     "${WITH_GPU}" WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  # )
  add_test(
    NAME test_netbuilder
    COMMAND
      ${CMAKE_COMMAND} -E env
      PYTHONPATH=${CMAKE_BINARY_DIR}:${CMAKE_BINARY_DIR}/python/cinn:$ENV{PYTHONPATH}
      python3 ${CMAKE_CURRENT_SOURCE_DIR}/test_netbuilder.py "${WITH_GPU}"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
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

add_test(
  NAME test_cinn_op_benchmark
  COMMAND
    ${CMAKE_COMMAND} -E env
    PYTHONPATH=${CMAKE_BINARY_DIR}:${CMAKE_BINARY_DIR}/python/cinn:$ENV{PYTHONPATH}
    python3 ${CMAKE_CURRENT_SOURCE_DIR}/test_op_benchmark.py "${WITH_GPU}"
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

if(WITH_CUDNN)
  add_test(
    NAME test_cinn_fake_resnet
    COMMAND
      ${CMAKE_COMMAND} -E env
      PYTHONPATH=${CMAKE_BINARY_DIR}:${CMAKE_BINARY_DIR}/python/cinn:$ENV{PYTHONPATH}
      python3 ${CMAKE_CURRENT_SOURCE_DIR}/test_resnet.py
      "${CMAKE_BINARY_DIR}/third_party/resnet_model" "${WITH_GPU}"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

  add_test(
    NAME test_cinn_real_resnet18
    COMMAND
      ${CMAKE_COMMAND} -E env
      PYTHONPATH=${CMAKE_BINARY_DIR}:${CMAKE_BINARY_DIR}/python/cinn:$ENV{PYTHONPATH}
      FLAGS_cinn_infer_model_version=1.0 python3
      ${CMAKE_CURRENT_SOURCE_DIR}/test_resnet18.py
      "${CMAKE_BINARY_DIR}/third_party/ResNet18" "${WITH_GPU}"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

  add_test(
    NAME test_cinn_real_mobilenetV2
    COMMAND
      ${CMAKE_COMMAND} -E env
      PYTHONPATH=${CMAKE_BINARY_DIR}:${CMAKE_BINARY_DIR}/python/cinn:$ENV{PYTHONPATH}
      FLAGS_cinn_infer_model_version=1.0 python3
      ${CMAKE_CURRENT_SOURCE_DIR}/test_mobilenetv2.py
      "${CMAKE_BINARY_DIR}/third_party/MobileNetV2" "${WITH_GPU}"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

  add_test(
    NAME test_cinn_real_efficientnet
    COMMAND
      ${CMAKE_COMMAND} -E env
      PYTHONPATH=${CMAKE_BINARY_DIR}:${CMAKE_BINARY_DIR}/python/cinn:$ENV{PYTHONPATH}
      FLAGS_cinn_infer_model_version=1.0 python3
      ${CMAKE_CURRENT_SOURCE_DIR}/test_efficientnet.py
      "${CMAKE_BINARY_DIR}/third_party/EfficientNet" "${WITH_GPU}"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

  add_test(
    NAME test_cinn_real_mobilenetV1
    COMMAND
      ${CMAKE_COMMAND} -E env
      PYTHONPATH=${CMAKE_BINARY_DIR}:${CMAKE_BINARY_DIR}/python/cinn:$ENV{PYTHONPATH}
      FLAGS_cinn_infer_model_version=1.0 python3
      ${CMAKE_CURRENT_SOURCE_DIR}/test_mobilenetv1.py
      "${CMAKE_BINARY_DIR}/third_party/MobilenetV1" "${WITH_GPU}"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

  add_test(
    NAME test_cinn_real_resnet50
    COMMAND
      ${CMAKE_COMMAND} -E env
      PYTHONPATH=${CMAKE_BINARY_DIR}:${CMAKE_BINARY_DIR}/python/cinn:$ENV{PYTHONPATH}
      FLAGS_cinn_infer_model_version=1.0 python3
      ${CMAKE_CURRENT_SOURCE_DIR}/test_resnet50.py
      "${CMAKE_BINARY_DIR}/third_party/ResNet50" "${WITH_GPU}"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

  add_test(
    NAME test_cinn_real_squeezenet
    COMMAND
      ${CMAKE_COMMAND} -E env
      PYTHONPATH=${CMAKE_BINARY_DIR}:${CMAKE_BINARY_DIR}/python/cinn:$ENV{PYTHONPATH}
      FLAGS_cinn_infer_model_version=1.0 python3
      ${CMAKE_CURRENT_SOURCE_DIR}/test_squeezenet.py
      "${CMAKE_BINARY_DIR}/third_party/SqueezeNet" "${WITH_GPU}"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

  add_test(
    NAME test_paddle_model_convertor
    COMMAND
      ${CMAKE_COMMAND} -E env
      PYTHONPATH=${CMAKE_BINARY_DIR}:${CMAKE_BINARY_DIR}/python/cinn:$ENV{PYTHONPATH}
      FLAGS_cinn_infer_model_version=1.0 python3
      ${CMAKE_CURRENT_SOURCE_DIR}/test_paddle_model_convertor.py --path
      "${CMAKE_BINARY_DIR}/third_party/resnet_model_1"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
endif()

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

  if(WITH_CUDNN)
    add_test(
      NAME test_conv2d_op
      COMMAND
        ${CMAKE_COMMAND} -E env
        PYTHONPATH=${CMAKE_BINARY_DIR}:${CMAKE_BINARY_DIR}/python/cinn:$ENV{PYTHONPATH}
        python3 ${CMAKE_CURRENT_SOURCE_DIR}/ops/test_conv2d_op.py
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
  endif()

  foreach(op_test_name ${EXCLUDE_OP})
    list(REMOVE_ITEM CINN_OP_TEST ops/${op_test_name}.py)
  endforeach()

  foreach(op_test_name ${CINN_OP_TEST})
    string(REGEX REPLACE ".py" "" op_test_name ${op_test_name})
    add_test(
      NAME ${op_test_name}
      COMMAND
        ${CMAKE_COMMAND} -E env
        PYTHONPATH=${CMAKE_BINARY_DIR}:${CMAKE_BINARY_DIR}/python/cinn:$ENV{PYTHONPATH}
        python3 ${CMAKE_CURRENT_SOURCE_DIR}/${op_test_name}.py
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
  endforeach()

  # test op mapper
  file(
    GLOB CINN_OP_MAPPER_TEST
    RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "op_mappers/test_*.py")
  set(EXCLUDE_OP_MAPPER test_mul_op test_conv2d_op)

  if(WITH_CUDNN)
    add_test(
      NAME test_mul_op_mapper
      COMMAND
        ${CMAKE_COMMAND} -E env
        PYTHONPATH=${CMAKE_BINARY_DIR}:${CMAKE_BINARY_DIR}/python/cinn:$ENV{PYTHONPATH}
        python3 ${CMAKE_CURRENT_SOURCE_DIR}/op_mappers/test_mul_op.py
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

    add_test(
      NAME test_conv2d_op_mapper
      COMMAND
        ${CMAKE_COMMAND} -E env
        PYTHONPATH=${CMAKE_BINARY_DIR}:${CMAKE_BINARY_DIR}/python/cinn:$ENV{PYTHONPATH}
        python3 ${CMAKE_CURRENT_SOURCE_DIR}/op_mappers/test_conv2d_op.py
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
  endif()

  foreach(op_mapper_test_name ${EXCLUDE_OP_MAPPER})
    list(REMOVE_ITEM CINN_OP_MAPPER_TEST op_mappers/${op_mapper_test_name}.py)
  endforeach()

  foreach(op_mapper_test_name ${CINN_OP_MAPPER_TEST})
    string(REGEX REPLACE ".py" "" op_mapper_test_name ${op_mapper_test_name})
    add_test(
      NAME "${op_mapper_test_name}_mapper"
      COMMAND
        ${CMAKE_COMMAND} -E env
        PYTHONPATH=${CMAKE_BINARY_DIR}:${CMAKE_BINARY_DIR}/python/cinn:$ENV{PYTHONPATH}
        python3 ${CMAKE_CURRENT_SOURCE_DIR}/${op_mapper_test_name}.py
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
  endforeach()

  # test pass test
  file(
    GLOB CINN_PASS_TEST
    RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "passes/test_*.py")

  foreach(pass_test_name ${EXCLUDE_PASS})
    list(REMOVE_ITEM CINN_PASS_TEST passes/${pass_test_name}.py)
  endforeach()

  foreach(pass_test_name ${CINN_PASS_TEST})
    string(REGEX REPLACE ".py" "" pass_test_name ${pass_test_name})
    add_test(
      NAME ${pass_test_name}
      COMMAND
        ${CMAKE_COMMAND} -E env
        PYTHONPATH=${CMAKE_BINARY_DIR}:${CMAKE_BINARY_DIR}/python/cinn:$ENV{PYTHONPATH}
        python3 ${CMAKE_CURRENT_SOURCE_DIR}/${pass_test_name}.py
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
  endforeach()

  file(
    GLOB CINN_FUSION_TEST
    RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "fusion/test_*.py")

  foreach(fusion_test_name ${EXCLUDE_FUSION})
    list(REMOVE_ITEM CINN_FUSION_TEST fusion/${fusion_test_name}.py)
  endforeach()

  foreach(fusion_test_name ${CINN_FUSION_TEST})
    string(REGEX REPLACE ".py" "" fusion_test_name ${fusion_test_name})
    add_test(
      NAME ${fusion_test_name}
      COMMAND
        ${CMAKE_COMMAND} -E env
        PYTHONPATH=${CMAKE_BINARY_DIR}:${CMAKE_BINARY_DIR}/python/cinn:$ENV{PYTHONPATH}
        python3 ${CMAKE_CURRENT_SOURCE_DIR}/${fusion_test_name}.py
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
  endforeach()

endif()
