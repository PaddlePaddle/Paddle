# top api
add_subdirectory(api)
# top core components
add_subdirectory(core)
# top kernels for diff device
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
# top kernels for other tensor
add_subdirectory(selected_rows)
# top infershape
add_subdirectory(infershape)
# top public functors
add_subdirectory(module)
# top tests
add_subdirectory(tests)
