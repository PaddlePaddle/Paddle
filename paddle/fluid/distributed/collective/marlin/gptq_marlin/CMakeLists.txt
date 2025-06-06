set(SRCS kernel_fp16_ku4.cu)

cc_library(
  gptq_marlin_lib
  SRCS        ${SRCS}
  DEPS        
)

target_include_directories(gptq_marlin_lib
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_options(gptq_marlin_lib
  PRIVATE
    -Wno-reorder
)