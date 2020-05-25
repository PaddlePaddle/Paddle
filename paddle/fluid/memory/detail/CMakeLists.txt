cc_library(memory_block SRCS memory_block.cc memory_block_desc.cc meta_cache.cc DEPS place)

if(${WITH_GPU})
  nv_library(system_allocator SRCS system_allocator.cc DEPS gflags cpu_info gpu_info place)
else(${WITH_GPU})
  cc_library(system_allocator SRCS system_allocator.cc DEPS gflags cpu_info place)
endif(${WITH_GPU})

cc_test(system_allocator_test SRCS system_allocator_test.cc DEPS system_allocator)

cc_library(buddy_allocator SRCS buddy_allocator.cc DEPS memory_block system_allocator glog)

cc_test(buddy_allocator_test SRCS buddy_allocator_test.cc DEPS buddy_allocator)

if(WITH_TESTING)
  set_tests_properties(buddy_allocator_test PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE")
  add_custom_command(TARGET buddy_allocator_test POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/../../testdata/buddy_allocator_test_data ${CMAKE_CURRENT_BINARY_DIR}/buddy_allocator_test_data
  )
endif()
