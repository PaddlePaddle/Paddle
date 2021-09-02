add_subdirectory(tests)

cc_library(exp_allocator SRCS allocator.cc DEPS)
cc_library(exp_tensor_dtype SRCS data_type.cc DEPS)
cc_library(exp_tensor_storage SRCS storage.cc DEPS boost allocator)
cc_library(exp_dense_tensor SRCS dense_tensor.cc DEPS exp_tensor_dtype exp_tensor_storage)
