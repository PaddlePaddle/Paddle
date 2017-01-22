#################### test_config_parser #########################
add_test(NAME layers_test
  COMMAND ${PROJ_ROOT}/paddle/.set_python_path.sh -d ${PROJ_ROOT}/python/
        ${PYTHON_EXECUTABLE} ${PROJ_ROOT}/python/paddle/trainer_config_helpers/tests/layers_test.py
    WORKING_DIRECTORY ${PROJ_ROOT}/python/paddle)

add_test(NAME test_reset_hook
  COMMAND ${PROJ_ROOT}/paddle/.set_python_path.sh -d ${PROJ_ROOT}/python/
        ${PYTHON_EXECUTABLE} ${PROJ_ROOT}/python/paddle/trainer_config_helpers/tests/test_reset_hook.py
    WORKING_DIRECTORY ${PROJ_ROOT}/python/paddle)

add_paddle_exe(protobuf_equal
  ProtobufEqualMain.cpp)
add_test(NAME test_layerHelpers
  COMMAND
  ${PROJ_ROOT}/python/paddle/trainer_config_helpers/tests/configs/run_tests.sh ${PYTHON_EXECUTABLE}
  ${CMAKE_CURRENT_BINARY_DIR}/protobuf_equal
)
