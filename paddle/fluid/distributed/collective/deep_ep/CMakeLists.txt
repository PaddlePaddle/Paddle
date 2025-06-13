# DeepEP of 2025.05.19, commit: d5ca4495c0c068bc617102841b9322d378fac8ea
if(WITH_NVSHMEM)
  set(CMAKE_CUDA_FLAGS
      "${CMAKE_CUDA_FLAGS} -rdc=true --ptxas-options=--register-usage-level=10,--warn-on-local-memory-usage"
  )
  string(REPLACE "-gencode arch=compute_80,code=sm_80" "" CMAKE_CUDA_FLAGS
                 "${CMAKE_CUDA_FLAGS}")

  set(DEEPEP_KERNEL_SRCS kernels/intranode.cu kernels/runtime.cu
                         kernels/internode.cu kernels/internode_ll.cu)
  cc_library(
    deepep_kernels
    SRCS ${DEEPEP_KERNEL_SRCS}
    DEPS nvshmem cudadevrt)

  set_target_properties(deepep_kernels PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  set_target_properties(deepep_kernels PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS
                                                  ON)
else()
  set(DEEPEP_KERNEL_SRCS kernels/intranode.cu kernels/runtime.cu)
  cc_library(deepep_kernels SRCS ${DEEPEP_KERNEL_SRCS})
endif()

cc_library(
  deep_ep
  SRCS deep_ep.cpp src/event_pool.cc src/event.cc src/CUDAStream.cc
  DEPS phi common deepep_kernels)

set_target_properties(deep_ep PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
target_compile_options(deep_ep PRIVATE -Wno-reorder -Wno-unused-variable)
