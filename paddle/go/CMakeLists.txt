include_directories(${CMAKE_CURRENT_BINARY_DIR})

go_library(adder SRCS adder.go)

cc_test(cgo_test
        SRCS
        cgo_test.cc
        DEPS
        adder)
