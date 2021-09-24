# tcmpt api
add_subdirectory(api)
# tcmpt core components
add_subdirectory(core)
# tcmpt eigne functors, now paddle must compiled with eigen, but eigen just is
# one backend dtype, we should support cropping it for lite
add_subdirectory(eigen)
# tcmpt kernels for diff device
add_subdirectory(cpu)
if(WITH_GPU OR WITH_ROCM)
  # TODO(chenweihang): if hip can split from cuda impl, we should add hip dir
  add_subdirectory(cuda)
endif()
# TODO(chenweihang): migrate MKLDNN Kernel in the second phase of the project
if(WITH_MKLDNN)
  add_subdirectory(mkldnn)
endif()
# TODO(chenweihang): migrate NPU Kernel in the second phase of the project
if(WITH_ASCEND_CL)
  add_subdirectory(npu)
endif()
# TODO(chenweihang): migrate XPU Kernel in the second phase of the project
if(WITH_XPU)
  add_subdirectory(xpu)
endif()
# tcmpt infershape
add_subdirectory(infershape)
# TODO(xingfeng): tcmpt inner module API designed by a high-performance team
add_subdirectory(module)
# tcmpt tests
add_subdirectory(tests)
