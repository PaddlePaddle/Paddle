cc_library(cpu_allocator SRCS cpu_allocator.cc)

cc_test(cpu_allocator_test
    SRCS cpu_allocator_test.cc
    DEPS cpu_allocator gflags)

cc_library(paddle_memory DEPS cpu_allocator)
