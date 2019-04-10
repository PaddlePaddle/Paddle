if (NOT WITH_ARM_CPU)
  add_subdirectory(memory)
  add_subdirectory(platform)
endif(NOT WITH_ARM_CPU)

  add_subdirectory(framework)

if (NOT WITH_ARM_CPU)
  add_subdirectory(imperative)
  add_subdirectory(operators)
  add_subdirectory(string)
  add_subdirectory(recordio)
  add_subdirectory(pybind)
  
  add_subdirectory(train)
endif(NOT WITH_ARM_CPU)
# NOTE: please add subdirectory inference at last.
add_subdirectory(inference)
