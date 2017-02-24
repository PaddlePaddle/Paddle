add_test(NAME topology_test
        COMMAND ${PROJ_ROOT}/paddle/.set_python_path.sh -d ${PROJ_ROOT}/python/
        ${PYTHON_EXECUTABLE} ${PROJ_ROOT}/python/paddle/v2/tests/topology_test.py
        WORKING_DIRECTORY ${PROJ_ROOT}/python/paddle)
