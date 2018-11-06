add_subdirectory(memory)
add_subdirectory(platform)
add_subdirectory(framework)
add_subdirectory(operators)
add_subdirectory(string)

if (NOT WIN32)
add_subdirectory(pybind)
add_subdirectory(recordio)
endif(NOT WIN32)

# NOTE: please add subdirectory inference at last.
add_subdirectory(inference)
add_subdirectory(train)
