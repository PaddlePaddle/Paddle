#include "../cutlass_dual_gemm.h"

template<>
__global__ void DualKernel<cutlass::gemm::kernel::DualGemm<cutlass::half_t, 
                                                           float, 
                                                           false, 
                                                           cutlass::epilogue::thread::SiLu, 
                                                           cutlass::arch::Sm80>>(
    cutlass::gemm::kernel::DualGemm<cutlass::half_t, float, false, 
    cutlass::epilogue::thread::SiLu, cutlass::arch::Sm80>::Params params){
    using Operator = cutlass::gemm::kernel::DualGemm<cutlass::half_t, 
                                                           float, 
                                                           false, 
                                                           cutlass::epilogue::thread::SiLu, 
                                                           cutlass::arch::Sm80>; 
    Operator op;
    op(params);
}