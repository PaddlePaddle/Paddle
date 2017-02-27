add_test(NAME test_v2_layer
        COMMAND ${PROJ_ROOT}/paddle/.set_python_path.sh -d ${PROJ_ROOT}/python/
        ${PYTHON_EXECUTABLE} ${PROJ_ROOT}/python/paddle/v2/tests/test_layer.py
        WORKING_DIRECTORY ${PROJ_ROOT}/python/paddle)

add_test(NAME test_v2_api
        COMMAND bash ${PROJ_ROOT}/python/paddle/v2/tests/run_tests.sh ${PYTHON_EXECUTABLE})

add_test(NAME topology_test
        COMMAND ${PROJ_ROOT}/paddle/.set_python_path.sh -d ${PROJ_ROOT}/python/
        ${PYTHON_EXECUTABLE} ${PROJ_ROOT}/python/paddle/v2/tests/topology_test.py
        WORKING_DIRECTORY ${PROJ_ROOT}/python/paddle)
