
#include <vector>
#include "paddle/extension.h"
#include "generated/aot/matmul/matmul_fp16.h"
#include <cuda_fp16.h>

std::vector<paddle::Tensor> TritonMatMul(const paddle::Tensor& a, 
                                              const paddle::Tensor& b) {

    int m = a.shape()[0];
    int n = b.shape()[1];
    int k = a.shape()[1];
    auto c_out = paddle::full({m, n}, 0, a.dtype(), a.place());

    auto dev_a = a.data<phi::dtype::float16>();
    auto dev_b = b.data<phi::dtype::float16>();
    auto dev_c = c_out.data<phi::dtype::float16>();

    auto status = matmul_kernel_fp16(a.stream(), 
    (CUdeviceptr)(dev_a), (CUdeviceptr)(dev_b), (CUdeviceptr)(dev_c),
    m,n,k,
    k,1,
    n,1,
    n,1,0) ;

    assert(status == CUDA_SUCCESS);

    return {c_out};                                                            
}

std::vector<std::vector<int64_t>> TritonMatMulInferShape(const std::vector<int64_t>& a_shape,
                                                              const std::vector<int64_t>& b_shape) {
    return {{a_shape[0], b_shape[1]}};
}

std::vector<paddle::DataType> TritonMatMulInferDtype(const paddle::DataType& A_dtype,
                                                        const paddle::DataType& B_dtype) {
    return {A_dtype};
}

PD_BUILD_OP(triton_matmul)
    .Inputs({"A", "B"})
    .Outputs({"C"})
    .SetKernelFn(PD_KERNEL(TritonMatMul))
    .SetInferShapeFn(PD_INFER_SHAPE(TritonMatMulInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(TritonMatMulInferDtype));
