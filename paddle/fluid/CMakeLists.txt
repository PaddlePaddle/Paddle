# Adapt to custom op mechanism: Include the header files related to the data type
# to avoid exposing the path of the underlying file
include_directories(${PADDLE_SOURCE_DIR}/paddle/fluid/platform)

add_subdirectory(memory)
add_subdirectory(platform)
add_subdirectory(distributed)
add_subdirectory(framework)
add_subdirectory(imperative)
add_subdirectory(operators)
add_subdirectory(string)
add_subdirectory(pybind)

# NOTE: please add subdirectory inference at last.
add_subdirectory(inference)
