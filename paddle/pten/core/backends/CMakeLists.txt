add_subdirectory(host)

message(WARNING "WITH_MKLDNN is ${WITH_MKLDNN}")
if(WITH_MKLDNN)
  set(MKLDNN_CTX_DEPS mkldnn)
else()
  set(MKLDNN_CTX_DEPS)
endif()

# TODO(support mkldnn context)
set(backends_context host_context ${MKLDNN_CTX_DEPS})

if (WITH_GPU OR WITH_ROCM)
  add_subdirectory(gpu)
  set(backends_context ${backends_context} ${gpu_context} PARENT_SCOPE)
endif()

if(WITH_XPU)
  add_subdirectory(xpu)
  set(backends_context ${backends_context} xpu_context PARENT_SCOPE)
endif()
