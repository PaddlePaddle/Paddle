cmake_minimum_required(VERSION 3.0)

add_executable(main main.c)
add_dependencies(main paddle_pserver_cclient)

if(APPLE)
  set(CMAKE_EXE_LINKER_FLAGS "-framework CoreFoundation -framework Security")
endif()

if(PROJ_ROOT)
  include_directories(${CMAKE_BINARY_DIR}/go/pserver/cclient/)
  target_link_libraries(main ${CMAKE_BINARY_DIR}/go/pserver/cclient/libpaddle_pserver_cclient.a pthread)
else(PROJ_ROOT)
  include_directories(${CMAKE_BINARY_DIR})
  target_link_libraries(main ${CMAKE_BINARY_DIR}/libpaddle_pserver_cclient.a pthread)
endif(PROJ_ROOT)
