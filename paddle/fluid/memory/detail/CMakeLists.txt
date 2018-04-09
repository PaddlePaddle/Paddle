cc_library(memory_block SRCS memory_block.cc memory_block_desc.cc meta_cache.cc)

if(${WITH_GPU})
  nv_library(system_allocator SRCS system_allocator.cc DEPS gflags cpu_info gpu_info)
elseif (WITH_AMD_GPU)
  hip_library(system_allocator SRCS system_allocator.cc DEPS gflags cpu_info gpu_info)
else(${WITH_GPU})
  cc_library(system_allocator SRCS system_allocator.cc DEPS gflags cpu_info)
endif(${WITH_GPU})

cc_test(system_allocator_test SRCS system_allocator_test.cc DEPS system_allocator)

cc_library(buddy_allocator SRCS buddy_allocator.cc DEPS memory_block system_allocator glog)
