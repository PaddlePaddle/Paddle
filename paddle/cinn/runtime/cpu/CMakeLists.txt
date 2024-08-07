core_gather_headers()

gather_srcs(cinnapi_src SRCS host_intrinsics.cc thread_backend.cc)

if(WITH_MKL_CBLAS)
  gather_srcs(cinnapi_src SRCS mkl_math.cc cblas.cc)
  if(WITH_ONEDNN)
    gather_srcs(cinnapi_src SRCS onednn_math.cc)
  endif()
endif()
