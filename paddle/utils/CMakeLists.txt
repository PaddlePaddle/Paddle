# The utilities for paddle

file(GLOB UTIL_HEADERS . *.h)
file(GLOB UTIL_SOURCES . *.cpp)
if(APPLE)
    file(GLOB UTIL_ARCH_SOURCES . arch/osx/*.cpp)
else()
    file(GLOB UTIL_ARCH_SOURCES . arch/linux/*.cpp)
endif()
add_library(paddle_utils STATIC
        ${UTIL_SOURCES}
        ${UTIL_ARCH_SOURCES})
add_style_check_target(paddle_utils ${UTIL_HEADERS})
add_style_check_target(paddle_utils ${UTIL_SOURCES}
    ${UTIL_ARCH_SOURCES})
add_dependencies(paddle_utils gen_proto_cpp)
if(WITH_TESTING)
    add_subdirectory(tests)
endif()
