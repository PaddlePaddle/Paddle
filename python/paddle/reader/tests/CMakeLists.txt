add_test(NAME reader_decorator_test
  COMMAND ${PROJ_ROOT}/paddle/.set_python_path.sh -d ${PROJ_ROOT}/python/
        ${PYTHON_EXECUTABLE} ${PROJ_ROOT}/python/paddle/reader/decorator_test.py
    WORKING_DIRECTORY ${PROJ_ROOT}/python/paddle)
