core_gather_headers()

gather_srcs(cinnapi_src SRCS header_generator.cc nvrtc_util.cc)

cinn_nv_test(test_nvrtc_util SRCS nvrtc_util_test.cc DEPS cinncore)
