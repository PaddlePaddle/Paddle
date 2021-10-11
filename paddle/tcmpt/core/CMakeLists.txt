add_subdirectory(tests)
add_subdirectory(utils)

cc_library(tcmpt_core SRCS dense_tensor.cc tensor_base.cc allocator.cc data_type.cc storage.cc )
