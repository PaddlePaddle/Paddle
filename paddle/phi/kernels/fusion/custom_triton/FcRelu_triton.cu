#include <vector>
#include "paddle/extension.h"
#include "generated/aot/FcRelu/fp16/FcRelu_fp16.h"
#include <cuda_fp16.h>
#include <cmath>  
#include <iostream>

std::vector<paddle::Tensor> TritonFcRelu(const paddle::Tensor& a, 
                                            const paddle::Tensor& b,
                                            const paddle::Tensor& bias) {

    int m = a.shape()[0];
    int n = b.shape()[1];
    int k = a.shape()[1];
    auto c_out = paddle::full({m, n}, 0, a.dtype(), a.place());

    auto dev_a = a.data<phi::dtype::float16>();
    auto dev_b = b.data<phi::dtype::float16>();
    auto dev_bias = bias.data<phi::dtype::float16>();
    auto dev_c = c_out.data<phi::dtype::float16>();

    auto status = FcRelu_kernel_fp16(c_out.stream(), 
    (CUdeviceptr)(dev_a), (CUdeviceptr)(dev_b), 
    (CUdeviceptr)(dev_c), (CUdeviceptr)(dev_bias), 
    m,n,k,
    k,1,
    n,1,
    n,1,0);

    assert(status == CUDA_SUCCESS);

    return {c_out};                                                            
}
std::vector<std::vector<int64_t>> TritonFcReluInferShape(const std::vector<int64_t>& a_shape,
                                                              const std::vector<int64_t>& b_shape, 
                                                              const std::vector<int64_t>& bias_shape) {
    return {a_shape, b_shape, bias_shape};
}

std::vector<paddle::DataType> TritonFcReluInferDtype(const paddle::DataType& a_dtype,
                                                        const paddle::DataType& b_dtype,
                                                        const paddle::DataType& bias_dtype) {
    return {a_dtype};
}

PD_BUILD_OP(triton_FcRelu)
    .Inputs({"a", "b", "bias"})
    .Outputs({"c"})
    .SetKernelFn(PD_KERNEL(TritonFcRelu))
    .SetInferShapeFn(PD_INFER_SHAPE(TritonFcReluInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(TritonFcReluInferDtype));
