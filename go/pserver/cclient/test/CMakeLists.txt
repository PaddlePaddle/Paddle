cmake_minimum_required(VERSION 3.0)

include_directories(${CMAKE_BINARY_DIR})
include_directories(${CMAKE_BINARY_DIR}/go/pserver/cclient/)


message(STATUS "CMAKE_CURRENT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..")
message(STATUS "CMAKE_BINARY_DIR: ${CMAKE_BINARY_DIR}")

add_executable(main main.c)
add_dependencies(main paddle_cclient)

if(APPLE)
  set(CMAKE_EXE_LINKER_FLAGS "-framework CoreFoundation -framework Security")
endif()
target_link_libraries(main ${CMAKE_BINARY_DIR}/go/pserver/cclient/libpaddle_cclient.a)

