set(AVX_SOURCES
    src/hl_math.cc
    src/hl_avx_functions.cc
)

if(WITH_AVX)
    set(CUDA_SOURCES
        src/hl_time.cc
        src/hl_cpu_functions.cc
        ${AVX_SOURCES})
else()
    set(CUDA_SOURCES
        src/hl_time.cc
        src/hl_cpu_functions.cc)
endif()

set(CUDA_CXX_WITH_GPU_SOURCES
    src/hl_cuda_cublas.cc
    src/hl_cuda_cudnn.cc
    src/hl_cuda_device.cc)

if(WITH_GPU)
    set(CUDA_CXX_SOURCES
        src/hl_warpctc_wrap.cc
        ${CUDA_CXX_WITH_GPU_SOURCES})

    set_source_files_properties(${CUDA_CXX_SOURCES}
                                PROPERTIES COMPILE_FLAGS "-D__NVCC__")
else()
    if (NOT MOBILE_INFERENCE)
    set(CUDA_CXX_SOURCES src/hl_warpctc_wrap.cc)
    endif()
endif()

set(CUDA_CU_SOURCES
    src/hl_perturbation_util.cu
    src/hl_cuda_aggregate.cu
    src/hl_cuda_matrix.cu
    src/hl_cuda_sparse.cu
    src/hl_cuda_cnn.cu
    src/hl_cuda_lstm.cu
    src/hl_top_k.cu
    src/hl_batch_transpose.cu
    src/hl_batch_norm.cu
    src/hl_cuda_sequence.cu
    src/hl_table_apply.cu)

set(CUDA_HEADERS
    include/hl_time.h
    include/hl_warpctc_wrap.h
    include/hl_sequence.h
    include/hl_cuda_cublas.h
    include/hl_batch_transpose.h
    include/hl_avx_functions.h
    include/hl_sparse.h
    include/hl_functions.h
    include/hl_cuda_cudnn.h
    include/hl_activation_functions.h
    include/hl_base.h
    include/stub/hl_cuda_cudnn_stub.h
    include/stub/hl_cuda_stub.h
    include/stub/hl_cuda_cublas_stub.h
    include/stub/hl_cnn_stub.h
    include/stub/hl_lstm_stub.h
    include/stub/hl_sequence_stub.h
    include/stub/hl_aggregate_stub.h
    include/stub/hl_sparse_stub.h
    include/stub/hl_matrix_stub.h
    include/hl_aggregate.h
    include/hl_cuda.h
    include/hl_lstm.h
    include/hl_table_apply.h
    include/hl_gpu.h
    include/hl_top_k.h
    include/hl_matrix.h
    include/hl_cnn.h)

if(WITH_GPU)
    cuda_add_library(paddle_cuda
        ${CUDA_SOURCES}
        ${CUDA_CU_SOURCES}
        ${CUDA_CXX_SOURCES})
else()
    add_library(paddle_cuda
                ${CUDA_SOURCES}
                ${CUDA_CXX_SOURCES})
endif()

add_dependencies(paddle_cuda paddle_proto ${external_project_dependencies})
