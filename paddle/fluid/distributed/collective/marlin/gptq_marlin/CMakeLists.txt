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

add_deep_ep_library(gptq_marlin_lib kernel_fp16_ku4.cu)

target_include_directories(gptq_marlin_lib
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_options(gptq_marlin_lib
  PRIVATE -Wno-reorder
)