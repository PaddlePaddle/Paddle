if(${WITH_GPU})
  nv_library(system_allocator SRCS system_allocator.cc DEPS gflags)
  nv_test(system_allocator_test
    SRCS system_allocator_test.cc
    DEPS system_allocator gpu_info gflags)
else(${WITH_GPU})
  cc_library(system_allocator SRCS system_allocator.cc DEPS gflags)
  cc_test(system_allocator_test SRCS system_allocator_test.cc DEPS system_allocator gflags)
endif(${WITH_GPU})

cc_library(metadata SRCS metadata.cc)
cc_library(buddy_allocator SRCS buddy_allocator.cc)
