enable_language(CUDA)

set(CUDA_SRCS
  kernel_fp16_ku4.cu
)

add_library(gptq_marlin_lib STATIC
  ${CUDA_SRCS}
)

# 包含当前目录的头文件
target_include_directories(gptq_marlin_lib
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# 强制 C++/CUDA 标准并启用可分离编译
set_target_properties(gptq_marlin_lib PROPERTIES
  POSITION_INDEPENDENT_CODE    ON
  CXX_STANDARD_REQUIRED        ON
  CXX_STANDARD                 17
  CUDA_STANDARD_REQUIRED       ON
  CUDA_STANDARD                17
  CUDA_SEPARABLE_COMPILATION   ON
)

# 针对 CUDA 源文件关闭 reorder 警告
target_compile_options(gptq_marlin_lib
  PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Wno-reorder>
)

# 链接 CUDA 运行时（若需要也可添加其他依赖库）
target_link_libraries(gptq_marlin_lib
  PUBLIC
    cudart
    cudadevrt
)
