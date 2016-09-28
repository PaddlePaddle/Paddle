#################### test_config_parser #########################
add_test(NAME layers_test
  COMMAND ${PROJ_ROOT}/paddle/.set_python_path.sh -d ${PROJ_ROOT}/python/
        python ${PROJ_ROOT}/python/paddle/trainer_config_helpers/tests/layers_test.py
    WORKING_DIRECTORY ${PROJ_ROOT}/python/paddle)

add_test(NAME test_layerHelpers
  COMMAND
  ${PROJ_ROOT}/python/paddle/trainer_config_helpers/tests/configs/run_tests.sh
)
