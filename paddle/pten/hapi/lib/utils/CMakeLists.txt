add_subdirectory(tests)

cc_library(pten_hapi_utils SRCS allocator.cc storage tensor_utils DEPS tensor_base pten_dense_tensor pten_utils)
