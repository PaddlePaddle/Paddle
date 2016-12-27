# Check package for each cmake option

if(WITH_GPU)
  find_package(CUDA REQUIRED)  # CUDA is required when use gpu
endif(WITH_GPU)

if(WITH_DOC)
  find_package(Sphinx REQUIRED)
  find_python_module(recommonmark REQUIRED)
endif(WITH_DOC)

if(WITH_SWIG_PY)
  find_python_module(wheel REQUIRED)  # package wheel
endif(WITH_SWIG_PY)
