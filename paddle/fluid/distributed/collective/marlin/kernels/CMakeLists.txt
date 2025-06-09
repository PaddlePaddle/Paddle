enable_language(CUDA)

set(CUDA_SRCS
  kernel_fp16_ku4.cu
)

add_subdirectory("../gptq_marlin" "gptq_marlin_lib")

cc_library(
  kernels_lib
  CUDASRCS    ${CUDA_SRCS}
  DEPS        gptq_marlin_lib
)

target_include_directories(kernels_lib
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(kernels_lib PROPERTIES
  CXX_STANDARD_REQUIRED ON
  CUDA_STANDARD_REQUIRED ON
  CXX_STANDARD 17
  CUDA_STANDARD 17
  CUDA_SEPARABLE_COMPILATION ON
)

target_compile_options(kernels_lib
  PRIVATE
    -Wno-reorder
)