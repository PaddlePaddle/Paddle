cc_library(cxx_api_lite SRCS cxx_api.cc DEPS scope_lite op_executor_lite host_kernels ops_lite optimizer_lite target_wrapper_host )
cc_test(test_cxx_api_lite SRCS cxx_api_test.cc DEPS cxx_api_lite model_parser_lite)

if(LITE_WITH_CUDA)
    cc_library(cxx_api_lite_cuda SRCS cxx_api.cc DEPS scope_lite op_executor_lite host_kernels ops_lite optimizer_lite target_wrapper_host target_wrapper_cuda kernels_cuda)
    nv_test(test_cxx_api_lite_cuda SRCS cxx_api_test.cc DEPS cxx_api_lite_cuda model_parser_lite)
endif()
