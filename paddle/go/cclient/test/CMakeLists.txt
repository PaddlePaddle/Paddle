cmake_minimum_required(VERSION 3.0)

include_directories(/env/gopath/src/github.com/PaddlePaddle/Paddle/paddle/go/cclient/build/)

add_executable(main main.c)
add_dependencies(main client)
set (CMAKE_EXE_LINKER_FLAGS "-pthread")
target_link_libraries(main /env/gopath/src/github.com/PaddlePaddle/Paddle/paddle/go/cclient/build/libclient.a) # ${GTEST_LIBRARIES})
