set(SRCS kernel_fp16_ku4.cu)

add_subdirectory("../gptq_marlin" "gptq_marlin_lib")

cc_library(
  kernels_lib
  SRCS        ${SRCS}
  DEPS        gptq_marlin_lib
)

target_include_directories(kernels_lib
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_options(kernels_lib
  PRIVATE
    -Wno-reorder
)