cmake_minimum_required(VERSION 3.0)

include_directories(${CMAKE_BINARY_DIR})

add_executable(main main.c)
add_dependencies(main client)
set (CMAKE_EXE_LINKER_FLAGS "-pthread")
target_link_libraries(main ${CMAKE_BINARY_DIR}/libclient.a)
