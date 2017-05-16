cmake_minimum_required(VERSION 3.0)

include_directories(/env/gopath/src/github.com/PaddlePaddle/Paddle/paddle/go/pserver/lib/build/client)

add_executable(main main.c)
add_dependencies(main client)
set (CMAKE_EXE_LINKER_FLAGS "-pthread")
target_link_libraries(main /env/gopath/src/github.com/PaddlePaddle/Paddle/paddle/go/pserver/lib/build/client/libclient.a ${GTEST_LIBRARIES})
