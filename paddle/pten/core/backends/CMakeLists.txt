add_subdirectory(host)
set(backends_context ${host_context} PARENT_SCOPE)

if (WITH_GPU OR WITH_ROCM)
  add_subdirectory(gpu)
  set(backends_context ${backends_context} ${gpu_context} PARENT_SCOPE)
endif()
