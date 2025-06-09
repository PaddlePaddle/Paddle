cmake_minimum_required(VERSION 3.5)
project(my_kernels LANGUAGES CXX CUDA)

# 定义深度 EP 专用的 helper 函数
function(add_deep_ep_library target_name source_file)
  add_library(${target_name} STATIC ${source_file})
  set_target_properties(
    ${target_name}
    PROPERTIES POSITION_INDEPENDENT_CODE ON
               CXX_STANDARD_REQUIRED ON
               CUDA_STANDARD_REQUIRED ON
               CXX_STANDARD 17
               CUDA_STANDARD 17
               CUDA_SEPARABLE_COMPILATION ON)
  target_link_libraries(${target_name} PUBLIC cudart cudadevrt)
endfunction()

add_subdirectory("../gptq_marlin" "gptq_marlin_lib")

add_deep_ep_library(kernels_lib kernel_fp16_ku4.cu)

# 链接 gptq_marlin_lib 依赖
target_link_libraries(kernels_lib PUBLIC gptq_marlin_lib)

target_include_directories(kernels_lib
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_options(kernels_lib
  PRIVATE -Wno-reorder
)