core_gather_headers()

gather_srcs(cinnapi_src SRCS host_intrinsics.cc thread_backend.cc)

if(WITH_MKL_CBLAS)
  gather_srcs(cinnapi_src SRCS mkl_math.cc cblas.cc)
  if(WITH_MKLDNN)
    gather_srcs(cinnapi_src SRCS mkldnn_math.cc)
  endif()
endif()

cinn_cc_test(test_host_intrinsics SRCS host_intrinsics_test.cc DEPS cinncore)
if(WITH_MKL_CBLAS)
  if(NOT WITH_CUDA)
    cinn_cc_test(test_mkl_math SRCS mkl_math_test.cc mkl_math.cc DEPS cinncore)
  endif()

  if(WITH_MKLDNN)
    cinn_cc_test(test_mkldnn_math SRCS mkldnn_math_test.cc mkldnn_math.cc DEPS
                 cinncore)
  endif()
endif()
