add_subdirectory(lib)
cc_library(pten_api SRCS all.cc DEPS pten_function_api pten_bw_function_api manual_api sparse_api)
