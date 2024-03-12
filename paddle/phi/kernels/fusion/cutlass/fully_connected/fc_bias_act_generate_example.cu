// #include "cutlass/gemm/device/gemm_universal.h"
// #include "cutlass/epilogue/thread/linear_combination_leaky_relu.h"
// #include "cutlass/epilogue/thread/linear_combination_silu.h"
// #include "cutlass/epilogue/thread/linear_combination_bias_relu.h"
// #include "cutlass/epilogue/thread/linear_combination_sigmoid.h"
// #include "cutlass/util/device_memory.h"
// #include "paddle/phi/kernels/fusion/cutlass/fully_connected/fc_util.h"


// // namespace phi{
// // namespace fusion{
// // namespace cutlass_internal{

// cutlass::Status fc_bias_relu_sm80_half_1(const FcAllParams& params) {
//     /// CommonCutlassFcKernelDeclare
//     using DeviceKernalName = cutlass::gemm::device::GemmUniversal<
//         cutlass::half_t, cutlass::layout::RowMajor,
//         cutlass::half_t, cutlass::layout::RowMajor,
//         cutlass::half_t,cutlass::layout::RowMajor,
//         float,
//         cutlass::arch::OpClassTensorOp,
//         cutlass::arch::Sm80,
//         cutlass::gemm::GemmShape<256, 128, 32>,
//         cutlass::gemm::GemmShape<64, 64, 32>,
//         cutlass::gemm::GemmShape<16,8,16>,
//         cutlass::epilogue::thread::LinearCombinationRelu<cutlass::half_t, 8, float, float>,
//         cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
//         3,
//         8,
//         8,
//         cutlass::arch::OpMultiplyAdd    /// Operation performed by GEMM
//     >;

//     /// Arguments
//     cutlass::gemm::GemmCoord problem_size{params.m, params.n, params.k};
//     cutlass::half_t *input = (cutlass::half_t *)(params.input);
//     cutlass::half_t *weight = (cutlass::half_t *)(params.weight);
//     cutlass::half_t *bias = (cutlass::half_t *)(params.bias);
//     cutlass::half_t *output = (cutlass::half_t *)(params.output);
    
//     int64_t batch_stride_C = problem_size.n();
//     long lda = (long)params.lda;   
//     long ldb = (long)params.ldb;
//     long ldc_bias = 0;
//     long ldd = (long)params.ldd;

//     typename DeviceKernalName::Arguments arguments{
//         cutlass::gemm::GemmUniversalMode::kGemm,
//         problem_size,
//         1,
//         {
//             float(params.alpha),
//             float(params.beta)
//         },
//         input,
//         weight,
//         bias,
//         output,
//         problem_size.mk().product(),
//         problem_size.nk().product(),
//         batch_stride_C,
//         problem_size.mn().product(),
//         lda,
//         ldb,
//         ldc_bias,
//         ldd,
        
//     };

//     /// CommonCutlassFcKernelExecute
//     DeviceKernalName device_gemm;
//     size_t workspace_size = device_gemm.get_workspace_size(arguments);
//     cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

//     cutlass::Status status = device_gemm.can_implement(arguments);
//     CUTLASS_CHECK(status);
//     status = device_gemm.initialize(arguments, workspace.get());
//     CUTLASS_CHECK(status);
//     status = device_gemm();
//     CUTLASS_CHECK(status);
//     return status;
// }


// // }  // namespace cutlass_internal
// // }  // namespace fusion
// // }  // namespace phi