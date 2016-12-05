# unittest for common package

add_simple_unittest(test_ExecViaCpu)
add_simple_unittest(test_SIMDFunctions)
add_simple_unittest(test_SparseMatrix)

# TODO(yuyang18): Refactor TestUtil.cpp. Remove this cross module reference.
add_unittest(test_matrixCompare
    test_matrixCompare.cpp
    ../../gserver/tests/TestUtil.cpp)

add_simple_unittest(test_sparseMatrixCompare)
add_simple_unittest(test_perturbation)
add_simple_unittest(test_CpuGpuVector)
add_simple_unittest(test_Allocator)
add_simple_unittest(test_FPException)
add_simple_unittest(test_GpuProfiler)
add_simple_unittest(test_BaseMatrix)
add_simple_unittest(test_Matrix)
