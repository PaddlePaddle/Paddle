if(${WITH_GPU})
  nv_library(system_allocator SRCS system_allocator.cc DEPS gflags gpu_info)
else(${WITH_GPU})
  cc_library(system_allocator SRCS system_allocator.cc DEPS gflags)
endif(${WITH_GPU})
cc_test(system_allocator_test SRCS system_allocator_test.cc DEPS system_allocator)

cc_library(meta_data SRCS meta_data.cc)
cc_library(buddy_allocator SRCS buddy_allocator.cc)
