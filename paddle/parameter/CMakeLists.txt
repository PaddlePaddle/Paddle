# The utilities for paddle

file(GLOB PARAMETERS_HEADERS . *.h)
file(GLOB PARAMETERS_SOURCES . *.cpp)

add_library(paddle_parameter STATIC
        ${PARAMETERS_SOURCES})
add_style_check_target(paddle_parameter ${PARAMETERS_SOURCES})
add_style_check_target(paddle_parameter ${PARAMETERS_HEADERS})
add_dependencies(paddle_parameter gen_proto_cpp)
if(WITH_TESTING)
    add_subdirectory(tests)
endif()
