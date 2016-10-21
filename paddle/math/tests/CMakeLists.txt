# unittest for common package

add_simple_unittest(test_ExecViaCpu)
add_simple_unittest(test_SIMDFunctions)
add_simple_unittest(test_matrix)
add_simple_unittest(test_TrainingAlgorithm)

# TODO(yuyang18): Refactor TestUtil.cpp. Remove this cross module reference.
add_unittest(test_matrixCompare
    test_matrixCompare.cpp
    ../../gserver/tests/TestUtil.cpp)

add_simple_unittest(test_sparseMatrixCompare)
add_simple_unittest(test_perturbation)
add_simple_unittest(test_CpuGpuVector)
add_simple_unittest(test_Allocator)
if(WITH_GPU)
	CUDA_ADD_EXECUTABLE(test_Tensor test_Tensor.cu)
	link_paddle_test(test_Tensor)
    if(COMPILER_SUPPORT_CXX11)
        CUDA_ADD_EXECUTABLE(test_lazyAssign test_lazyAssign.cu)
        link_paddle_test(test_lazyAssign)
    endif()
else()
    compile_cu_as_cpp(test_Tensor.cu)
    add_unittest(test_Tensor test_Tensor.cu)
    compile_cu_as_cpp(test_lazyAssign.cu)
    add_unittest(test_lazyAssign test_lazyAssign.cu)
endif(WITH_GPU)

