if(${WITH_GPU})
  nv_test(cpu_allocator_test SRCS cpu_allocator_test.cc) # nv_test links CUDA, but
else(${WITH_GPU})
  cc_test(cpu_allocator_test SRCS cpu_allocator_test.cc) # cc_test doesn't.
endif(${WITH_GPU})
