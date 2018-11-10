#################### test_config_parser #########################
add_test(NAME layers_test
  COMMAND ${PADDLE_SOURCE_DIR}/paddle/.set_python_path.sh -d ${PADDLE_BINARY_DIR}/python/
        ${PYTHON_EXECUTABLE} ${PADDLE_SOURCE_DIR}/python/paddle/trainer_config_helpers/tests/layers_test.py
    WORKING_DIRECTORY ${PADDLE_SOURCE_DIR}/python/paddle)

add_test(NAME test_reset_hook
  COMMAND ${PADDLE_SOURCE_DIR}/paddle/.set_python_path.sh -d ${PADDLE_BINARY_DIR}/python/
        ${PYTHON_EXECUTABLE} ${PADDLE_SOURCE_DIR}/python/paddle/trainer_config_helpers/tests/test_reset_hook.py
    WORKING_DIRECTORY ${PADDLE_SOURCE_DIR}/python/paddle)

add_paddle_exe(protobuf_equal ProtobufEqualMain.cpp)
add_test(NAME test_layerHelpers
  COMMAND ${PADDLE_SOURCE_DIR}/paddle/.set_python_path.sh -d ${PADDLE_BINARY_DIR}/python/
  ${PADDLE_BINARY_DIR}/python/paddle/trainer_config_helpers/tests/configs/run_tests.sh ${PYTHON_EXECUTABLE}
  ${CMAKE_CURRENT_BINARY_DIR}/protobuf_equal
)
