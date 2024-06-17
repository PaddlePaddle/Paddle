core_gather_headers()

gather_srcs(cinnapi_src SRCS host_intrinsics.cc thread_backend.cc)

if(WITH_MKL_CBLAS)
  gather_srcs(cinnapi_src SRCS mkl_math.cc cblas.cc)
  if(WITH_ONEDNN)
    gather_srcs(cinnapi_src SRCS onednn_math.cc)
  endif()
endif()

cinn_cc_test(test_host_intrinsics SRCS host_intrinsics_test.cc DEPS cinncore)
if(WITH_MKL_CBLAS)
  if(NOT WITH_CUDA)
    cinn_cc_test(test_mkl_math SRCS mkl_math_test.cc mkl_math.cc DEPS cinncore)
  endif()

  if(WITH_ONEDNN)
    cinn_cc_test(test_onednn_math SRCS onednn_math_test.cc onednn_math.cc DEPS
                 cinncore)
  endif()
endif()
