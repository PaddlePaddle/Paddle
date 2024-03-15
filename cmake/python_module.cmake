# Find if a Python module is installed
# Found at http://www.cmake.org/pipermail/cmake/2011-January/041666.html
# To use do: find_python_module(PyQt4 REQUIRED)
function(find_python_module module)
  string(TOUPPER ${module} module_upper)
  if(NOT PY_${module_upper})
    if(ARGC GREATER 1 AND ARGV1 STREQUAL "REQUIRED")
      set(${module}_FIND_REQUIRED TRUE)
    else()
      set(${module}_FIND_REQUIRED FALSE)
    endif()
    # A module's location is usually a directory, but for binary modules
    # it's a .so file.
    execute_process(
      COMMAND
        "${PYTHON_EXECUTABLE}" "-c"
        "import re, ${module}; print(re.compile('/__init__.py.*').sub('',${module}.__file__))"
      RESULT_VARIABLE _${module}_status
      OUTPUT_VARIABLE _${module}_location
      ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT _${module}_status)
      set(PY_${module_upper}
          ${_${module}_location}
          CACHE STRING "Location of Python module ${module}")
    endif()
  endif()
  find_package_handle_standard_args(PY_${module} DEFAULT_MSG PY_${module_upper})
  if(NOT PY_${module_upper}_FOUND AND ${module}_FIND_REQUIRED)
    message(FATAL_ERROR "python module ${module} is not found")
  endif()

  execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" "-c"
            "import sys, ${module}; sys.stdout.write(${module}.__version__)"
    OUTPUT_VARIABLE _${module}_version
    RESULT_VARIABLE _${module}_status
    ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT _${module}_status)
    set(PY_${module_upper}_VERSION
        ${_${module}_version}
        CACHE STRING "Version of Python module ${module}")
  endif()

  set(PY_${module_upper}_FOUND
      ${PY_${module_upper}_FOUND}
      PARENT_SCOPE)
  set(PY_${module_upper}_VERSION
      ${PY_${module_upper}_VERSION}
      PARENT_SCOPE)
endfunction()

function(check_py_version py_version)
  string(REPLACE "." ";" version_list ${py_version})
  list(LENGTH version_list version_list_len)
  if(version_list_len LESS 2)
    message(FATAL_ERROR "Please input Python version, eg:3.8 or 3.9 and so on")
  endif()

  list(GET version_list 0 version_major)
  list(GET version_list 1 version_minor)

  if((version_major GREATER_EQUAL 3) AND (version_minor GREATER_EQUAL 7))

  else()
    message(FATAL_ERROR "Paddle only support Python version >=3.8 now!")
  endif()
endfunction()
