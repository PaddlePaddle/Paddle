cmake_minimum_required(VERSION 3.0)

include_directories(${CMAKE_BINARY_DIR})

add_executable(recordio_test test.c)
add_dependencies(recordio_test recordio)
set (CMAKE_EXE_LINKER_FLAGS "-pthread")
target_link_libraries(recordio_test ${CMAKE_BINARY_DIR}/librecordio.a)
