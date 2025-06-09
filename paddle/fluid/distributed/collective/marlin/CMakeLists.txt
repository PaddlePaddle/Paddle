enable_language(CUDA)

set(CUDA_SRCS ops.cu)
add_subdirectory(kernels)

cc_library(
  marlin
  CUDA_SRCS ${CUDA_SRCS}
  DEPS phi common kernels_lib)

target_include_directories(marlin
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(marlin PROPERTIES
  CXX_STANDARD_REQUIRED ON
  CUDA_STANDARD_REQUIRED ON
  CXX_STANDARD 17
  CUDA_STANDARD 17
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_SEPARABLE_COMPILATION ON
)

# Link CPP and CUDA together
target_compile_options(marlin PRIVATE -Wno-reorder)
