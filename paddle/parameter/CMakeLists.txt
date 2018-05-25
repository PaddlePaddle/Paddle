# The utilities for paddle

file(GLOB PARAMETERS_HEADERS . *.h)
file(GLOB PARAMETERS_SOURCES . *.cpp)

add_library(paddle_parameter STATIC
        ${PARAMETERS_SOURCES})
add_dependencies(paddle_parameter paddle_proto ${external_project_dependencies})
if(WITH_TESTING)
    add_subdirectory(tests)
endif()
