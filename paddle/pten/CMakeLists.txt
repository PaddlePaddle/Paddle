# pten api
add_subdirectory(api)
# pten core components
add_subdirectory(core)
# pten kernels for diff device
add_subdirectory(cpu)
if(WITH_GPU)
  add_subdirectory(cuda)
endif()
if(WITH_ROCM)
  add_subdirectory(hip)
endif()
if(WITH_MKLDNN)
  add_subdirectory(mkldnn)
endif()
if(WITH_ASCEND_CL)
  add_subdirectory(npu)
endif()
if(WITH_XPU)
  add_subdirectory(xpu)
endif()
# pten kernels for other tensor
add_subdirectory(selected_rows)
# pten infershape and dtype
add_subdirectory(infershape)
add_subdirectory(inferdtype)
# pten public functors
add_subdirectory(module)
# pten tests
add_subdirectory(tests)
