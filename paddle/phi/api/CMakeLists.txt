add_subdirectory(lib)
cc_library(phi_api SRCS all.cc DEPS phi_function_api phi_bw_function_api sparse_api sparse_bw_api)
