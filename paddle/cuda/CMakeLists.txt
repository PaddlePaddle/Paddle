file(GLOB_RECURSE HEAD_SOURCES RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*.h")

set(SIMD_SOURCES
    src/simd_gru_op.cc
    src/simd_gru.cc
    src/simd_lstm_op.cc
    src/simd_lstm.cc)

add_library(avx_rnn ${SIMD_SOURCES})
set_target_properties(avx_rnn PROPERTIES COMPILE_FLAGS ${AVX_FLAG})

add_library(naive_rnn ${SIMD_SOURCES})
set_target_properties(naive_rnn PROPERTIES COMPILE_FLAGS ${SSE3_FLAG})

set(CPU_SOURCES
    src/hl_cpu_recurrent.cc
    src/hl_dso_loader.cc
    src/hl_warpctc_wrap.cc
    src/hl_time.cc)

set(CUDA_WRAP_SOURCES
    src/hl_cuda_cublas.cc
    src/hl_cuda_cudnn.cc
    src/hl_cuda_device.cc)
set_source_files_properties(${CUDA_WRAP_SOURCES} PROPERTIES COMPILE_FLAGS "-D__NVCC__")

set(CUDA_SOURCES		
    src/hl_perturbation_util.cu		
    src/hl_cuda_aggregate.cu		
    src/hl_cuda_matrix.cu		
    src/hl_cuda_sparse.cu		
    src/hl_cuda_cnn.cu		
    src/hl_cuda_lstm.cu		
    src/hl_top_k.cu		
    src/hl_batch_transpose.cu		
    src/hl_cuda_sequence.cu		
    src/hl_table_apply.cu)

if(WITH_GPU)
    cuda_add_library(paddle_cuda
                     ${CUDA_SOURCES}
                     ${CPU_SOURCES}
                     ${CUDA_WRAP_SOURCES})
else()
    add_library(paddle_cuda ${CPU_SOURCES})
endif()

target_link_libraries(paddle_cuda naive_rnn avx_rnn)

add_dependencies(paddle_cuda ${external_project_dependencies})

add_style_check_target(paddle_cuda
                       ${CPU_SOURCES}
                       ${HEAD_SOURCES}
                       ${SIMD_SOURCES}
                       ${CUDA_WRAP_SOURCES})
