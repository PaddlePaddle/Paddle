add_subdirectory(string)

if(NOT ((NOT WITH_PYTHON) AND ON_INFER))
  cc_library(
    pybind_util
    SRCS pybind.cc
    DEPS phi common python)
  if(WITH_PYTHON)
    target_include_directories(pybind_util PRIVATE ${PYTHON_INCLUDE_DIRS})
  endif()
endif()
cc_library(md5 SRCS md5.cc)
