add_test(NAME reader_decorator_test
  COMMAND ${PROJ_ROOT}/paddle/.set_python_path.sh -d ${PROJ_ROOT}/python/
        ${PYTHON_EXECUTABLE} ${PROJ_ROOT}/python/paddle/reader/tests/decorator_test.py
    WORKING_DIRECTORY ${PROJ_ROOT}/python/paddle)

add_test(NAME reader_creator_test
  COMMAND ${PROJ_ROOT}/paddle/.set_python_path.sh -d ${PROJ_ROOT}/python/
        ${PYTHON_EXECUTABLE} ${PROJ_ROOT}/python/paddle/reader/tests/creator_test.py
    WORKING_DIRECTORY ${PROJ_ROOT}/python/paddle)
