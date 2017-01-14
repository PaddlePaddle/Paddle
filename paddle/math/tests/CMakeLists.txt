# unittest for common package

add_simple_unittest(test_ExecViaCpu)
add_simple_unittest(test_SIMDFunctions)
add_simple_unittest(test_TrainingAlgorithm)
add_simple_unittest(test_SparseMatrix)
add_simple_unittest(test_RowBuffer)

# TODO(yuyang18): Refactor TestUtil.cpp. Remove this cross module reference.
add_unittest(test_matrixCompare
    test_matrixCompare.cpp)

add_simple_unittest(test_sparseMatrixCompare)
add_simple_unittest(test_perturbation)
add_simple_unittest(test_CpuGpuVector)
add_simple_unittest(test_Allocator)

if(WITH_GPU)
    CUDA_ADD_EXECUTABLE(test_Tensor test_Tensor.cu)
    link_paddle_test(test_Tensor)
    CUDA_ADD_EXECUTABLE(test_lazyAssign test_lazyAssign.cu)
    link_paddle_test(test_lazyAssign)
else()
    compile_cu_as_cpp(test_Tensor.cu)
    add_unittest(test_Tensor test_Tensor.cu)
    compile_cu_as_cpp(test_lazyAssign.cu)
    add_unittest(test_lazyAssign test_lazyAssign.cu)
endif(WITH_GPU)

add_simple_unittest(test_FPException)
add_simple_unittest(test_GpuProfiler)
add_simple_unittest(test_BaseMatrix)
add_simple_unittest(test_Matrix)
