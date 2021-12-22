add_subdirectory(lib)

cc_library(pten_api SRCS all.cc DEPS pten_function_api utils_api sparse_coo_tensor_util_api sparse_csr_tensor_utils_api)
