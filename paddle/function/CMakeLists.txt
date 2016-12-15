file(GLOB FUNCTION_HEADERS . *.h)

if(NOT WITH_GPU)
    file(GLOB FUNCTION_SOURCES . *.cpp)
    add_library(paddle_function STATIC ${FUNCTION_SOURCES})
else()
    file(GLOB FUNCTION_SOURCES . *.cpp *.cu)
    cuda_add_library(paddle_function ${FUNCTION_SOURCES})
endif()

add_style_check_target(paddle_function ${FUNCTION_SOURCES})
add_style_check_target(paddle_function ${FUNCTION_HEADERS})
