# Check package for each cmake option

if(WITH_GPU)
  find_package(CUDA REQUIRED)  # CUDA is required when use gpu
endif()

if(WITH_PYTHON)
  find_package(PythonLibs 2.6 REQUIRED)
  find_package(PythonInterp REQUIRED)
  find_package(NumPy REQUIRED)
endif()

if(WITH_STYLE_CHECK)
  find_package(PythonInterp REQUIRED)
endif()

find_package(Glog REQUIRED)

find_package(Gflags REQUIRED)

if(WITH_TESTING)
  find_package(GTest REQUIRED)
endif()

if(WITH_DOC)
  find_package(Sphinx REQUIRED)
  find_python_module(recommonmark REQUIRED)
endif()

if(WITH_SWIG_PY)
  if(NOT SWIG_FOUND)
    message(FATAL_ERROR "SWIG is not found. Please install swig or disable WITH_SWIG_PY")
  endif()
  find_python_module(wheel REQUIRED)  # package wheel
endif()

if(NOT M4_EXECUTABLE)
  message(FATAL_ERROR "Paddle need m4 to generate proto file.")
endif()
