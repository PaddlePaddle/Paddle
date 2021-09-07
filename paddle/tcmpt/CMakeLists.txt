# tcmpt api
add_subdirectory(api)
# tcmpt core components
add_subdirectory(core)
# tcmpt kernels for diff device
add_subdirectory(cpu)
if(WITH_GPU OR WITH_ROCM)
  add_subdirectory(cuda)
endif()
# TODO(chenweihang): if hip can split from cuda impl, we should add hip dir
if(WITH_MKLDNN)
  add_subdirectory(mkldnn)
endif()
if(WITH_ASCEND_CL)
  add_subdirectory(npu)
endif()
if(WITH_XPU)
  add_subdirectory(xpu)
endif()
# tcmpt infershape
add_subdirectory(infershape)
# tcmpt public functors
add_subdirectory(module)
# tcmpt tests
add_subdirectory(tests)
