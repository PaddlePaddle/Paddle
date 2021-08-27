cc_library(exp_allocator SRCS allocator.cc DEPS)
cc_library(exp_tensor_dtype SRCS data_type.cc DEPS)
cc_library(exp_dense_tensor SRCS dense_tensor.cc DEPS)
cc_library(exp_tensor_storage SRCS storage.cc DEPS boost allocator)
