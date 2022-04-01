add_subdirectory(memory)
add_subdirectory(platform)
add_subdirectory(distributed)
add_subdirectory(framework)
add_subdirectory(imperative)
add_subdirectory(operators)
add_subdirectory(pybind)

if(NOT ((NOT WITH_PYTHON) AND ON_INFER))
    add_subdirectory(eager)
endif()
# NOTE: please add subdirectory inference at last.
add_subdirectory(inference)
