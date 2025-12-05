# Find the Python NumPy package
# PYTHON_NUMPY_INCLUDE_DIR
# NUMPY_FOUND
# will be set by this script

if(NOT PYTHON_EXECUTABLE)
  if(NumPy_FIND_QUIETLY)
    find_package(PythonInterp QUIET)
  else()
    find_package(PythonInterp)
    set(_numpy_out 1)
  endif()
endif()

if(PYTHON_EXECUTABLE)
  # write a python script that finds the numpy path
  file(WRITE ${PROJECT_BINARY_DIR}/FindNumpyPath.py
       "try: import numpy; print(numpy.get_include())\nexcept:pass\n")

  # execute the find script
  execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" "FindNumpyPath.py"
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
    OUTPUT_VARIABLE NUMPY_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE)
elseif(_numpy_out)
  message(STATUS "Python executable not found.")
endif()

find_path(PYTHON_NUMPY_INCLUDE_DIR numpy/arrayobject.h
          HINTS "${NUMPY_PATH}" "${PYTHON_INCLUDE_PATH}")

if(PYTHON_NUMPY_INCLUDE_DIR)
  set(PYTHON_NUMPY_FOUND
      1
      CACHE INTERNAL "Python numpy found")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NumPy DEFAULT_MSG PYTHON_NUMPY_INCLUDE_DIR)
