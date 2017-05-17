include_directories(${CMAKE_CURRENT_BINARY_DIR})

go_library(adder SRCS adder.go)

# cc_test(cgo_test
#         SRCS
#         cgo_test.cc
#         DEPS
#         adder)
add_executable(cgo_test cgo_test.cc)
add_dependencies(cgo_test adder)
target_link_libraries(cgo_test ${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES} adder)
add_test(cgo_test cgo_test)
