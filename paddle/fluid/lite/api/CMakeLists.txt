set(cxx_api_lite_deps scope_lite optimizer_lite target_wrapper_host optimizer_lite model_parser_lite)
if(LITE_WITH_CUDA)
    set(cxx_api_lite_deps ${cxx_api_lite_deps} kernels_cuda)
    cc_library(cxx_api_lite_cuda SRCS cxx_api.cc DEPS ${cxx_api_lite_deps} target_wrapper_cuda)
    nv_test(test_cxx_api_lite_cuda SRCS cxx_api_test.cc DEPS cxx_api_lite_cuda)
endif()

cc_library(cxx_api_lite SRCS cxx_api.cc DEPS ${cxx_api_lite_deps} ${ops_lite})

set(light_api_deps
    scope_lite target_wrapper_host model_parser_lite)

if(LITE_WITH_CUDA)
    set(light_api_deps ${light_api_deps} target_wrapper_cuda)
endif()

cc_library(light_api_lite SRCS light_api.cc DEPS ${light_api_deps} ${ops_lite} ${host_kernels})

message(STATUS "get ops ${ops_lite}")
message(STATUS "get kernels ${host_kernels}")
lite_cc_test(test_cxx_api_lite SRCS cxx_api_test.cc DEPS cxx_api_lite model_parser_lite target_wrapper_host
  ${ops_lite} ${host_kernels})
lite_cc_test(test_light_api SRCS light_api_test.cc DEPS light_api_lite)

cc_binary(cxx_api_lite_bin SRCS cxx_api_bin.cc DEPS cxx_api_lite model_parser_lite target_wrapper_host ${ops_lite} ${host_kernels})
