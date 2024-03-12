#include <iostream>
#include <string>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/util/command_line.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"
#include "helper.h"

/// cutlass_tensorop_h16816gemm_128x128_32x4_nn_align8

using ElementA = cutlass::half_t;
using LayoutA = cutlass::layout::RowMajor;
constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;     // 什么意思，按128bit对齐？

using ElementB = cutlass::half_t;
using LayoutB = cutlass::layout::RowMajor;
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
// C
using ElementC = cutlass::half_t;
using LayoutC = cutlass::layout::RowMajor;
// constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
// D
using ElementD = cutlass::half_t;
using LayoutD = cutlass::layout::RowMajor;
// constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

using ElementAccumulator = cutlass::half_t;
// alpha
using ElementComputeEpilogue = ElementAccumulator;


using ArchTag = cutlass::arch::Sm80;
using OperatorClass = cutlass::arch::OpClassTensorOp;
using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;        // M N K
using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
constexpr int NumStages = 4;
// Epilog始终是行主序的 如果AB是列主序的，那么会使用BA
// Define the epilogue operation as LinearCombinationRelu. This is approximately equal to
//
//    d_ij = max(0, alpha * sum_k(a_ik * b_kj) + c_ij )
//
using EpilogueOp = cutlass::epilogue::thread::LinearCombinationRelu<
    ElementD, 
    128 / cutlass::sizeof_bits<ElementD>::value,
    ElementAccumulator,                                     // 内部累加
    ElementComputeEpilogue,                                 // 计算线性combination
    cutlass::epilogue::thread::ScaleType::NoBetaScaling>;

// 参考GEMM
using DeviceGemmReference = cutlass::reference::device::Gemm<
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementD, LayoutD,
    ElementComputeEpilogue,
    ElementComputeEpilogue>;

// GEMM Universal 经典数据并行 GemmIdentityThreadblockSwizzle要改一下？
using DeviceGemmBasic = cutlass::gemm::device::GemmUniversal<
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOp,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    NumStages,
    AlignmentA,
    AlignmentB>;

struct Options
{
  std::string               command_name;
  bool                      help;
  cutlass::gemm::GemmCoord  problem_size;
  float                     alpha;
  float                     beta;
  int                       split_k_factor;
  int                       avail_sms;
  bool                      reference_check;
  int                       iterations;

  cutlass::HostTensor<ElementA, LayoutA> tensor_A;
  cutlass::HostTensor<ElementB, LayoutB> tensor_B;
  cutlass::HostTensor<ElementD, LayoutD> tensor_C_bias;
  cutlass::HostTensor<ElementD, LayoutD> tensor_D;
  cutlass::HostTensor<ElementD, LayoutD> tensor_ref_D;

  Options(std::string command_name) :
    command_name(command_name),
    help(false),
    problem_size({2048, 2048, 2048}),
    alpha(1.0f),
    beta(0.0f),
    split_k_factor(1),
    avail_sms(-1),              // Number of device SMs to use is unlimited
    reference_check(true),
    iterations(10000)
  {}

  bool valid() const
  {
    return true;
  }

  void parse(int argc, char const **args)
  {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
    }

    cmd.get_cmd_line_argument("m", problem_size.m());
    cmd.get_cmd_line_argument("n", problem_size.n());
    cmd.get_cmd_line_argument("k", problem_size.k());
    cmd.get_cmd_line_argument("alpha", alpha);
    cmd.get_cmd_line_argument("beta", beta);
    cmd.get_cmd_line_argument("split", split_k_factor);
    cmd.get_cmd_line_argument("iterations", iterations);
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const
  {
    out
      << "Performs a GEMM computation.\n"
      << "\n"
      << "Options:\n"
      << "\n"
      << "  --help                      If specified, displays this usage statement.\n\n"
      << "  --m=<int>                   GEMM M dimension\n"
      << "  --n=<int>                   GEMM N dimension\n"
      << "  --k=<int>                   GEMM K dimension\n"
      << "  --alpha=<f32>               Epilogue scalar alpha\n"
      << "  --beta=<f32>                Epilogue scalar beta\n\n"
      << "  --split=<int>               Split-K factor to emulate\n\n"
      << "  --iterations=<int>          Number of profiling iterations to perform.\n\n";

    out
      << "\n\nExamples:\n\n"
      << "$ " << command_name << " --m=1024 --n=512 --k=1024 --alpha=2 --beta=0.707 \n\n";

    return out;
  }

  /// Compute performance in GFLOP/s
  double gflops(double runtime_s) const
  {
    // Two flops per multiply-add
    return 2.0 * double(problem_size.product()) / double(1.0e9) / runtime_s;
  }
};

typename DeviceGemmBasic::Arguments args_from_options(
    const DeviceGemmBasic &device_gemm,
    const Options &options,
    cutlass::HostTensor<ElementA, LayoutA> &tensor_A,
    cutlass::HostTensor<ElementB, LayoutB> &tensor_B,
    cutlass::HostTensor<ElementD, LayoutD> &tensor_C_bias,
    cutlass::HostTensor<ElementD, LayoutD> &tensor_D)
{
    return typename DeviceGemmBasic::Arguments(
        cutlass::gemm::GemmUniversalMode::kGemm,
        options.problem_size,
        options.split_k_factor,
        {
            ElementComputeEpilogue(options.alpha),
            ElementComputeEpilogue(options.beta)
        },
        tensor_A.device_data(),
        tensor_B.device_data(),
        tensor_C_bias.device_data(),
        tensor_D.device_data(),
        options.problem_size.mk().product(),
        options.problem_size.nk().product(),
        options.problem_size.n(),
        options.problem_size.mn().product(),
        tensor_A.layout().stride(0),
        tensor_B.layout().stride(0),
        0,
        tensor_D.layout().stride(0));
}

template <typename DeviceGemmT>
int run(std::string desc, Options &options){
    cutlass::reference::host::TensorFill(options.tensor_D.host_view());
    options.tensor_D.sync_device();

    DeviceGemmT device_gemm;
    auto arguments = args_from_options(device_gemm, options, options.tensor_A, options.tensor_B,
                                        options.tensor_C_bias, options.tensor_D);
    size_t workspace_size = DeviceGemmT::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    // Check the problem size is supported or not
    CUTLASS_CHECK(device_gemm.can_implement(arguments));
    // Initialize CUTLASS kernel with arguments and workspace pointer
    CUTLASS_CHECK(device_gemm.initialize(arguments, workspace.get()));
    // run
    CUTLASS_CHECK(device_gemm());
}

int main(int argc, const char** argv){
    Options options("ampere_gemm_universal");
    options.parse(argc, argv);
    // 创建矩阵
    options.tensor_A.resize(options.problem_size.mk());       // <- Create matrix A with dimensions M x K
    options.tensor_B.resize(options.problem_size.kn());       // <- Create matrix B with dimensions K x N
    options.tensor_C_bias.resize({1, options.problem_size.n()});
    options.tensor_D.resize(options.problem_size.mn());       // <- Create matrix D with dimensions M x N used to store output from CUTLASS kernel
    options.tensor_ref_D.resize(options.problem_size.mn());

    cutlass::reference::host::TensorFillRandomUniform(
        options.tensor_A.host_view(),
        1,
        ElementA(4),
        ElementA(-4),
        0);
    cutlass::reference::host::TensorFillRandomUniform(
        options.tensor_B.host_view(),
        1,
        ElementB(4),
        ElementB(-4),
        0);
    cutlass::reference::host::TensorFillRandomUniform(
        options.tensor_C_bias.host_view(),
        1,
        ElementD(4),
        ElementD(-4),
        0);
   
    options.tensor_A.sync_device();
    options.tensor_B.sync_device();
    options.tensor_C_bias.sync_device();

    /// 计算ref_D
    cutlass::reference::host::TensorFill(options.tensor_D.host_view());
    options.tensor_ref_D.sync_device();
    DeviceGemmReference gemm_reference;
    gemm_reference(
        options.problem_size,
        ElementComputeEpilogue(options.alpha),
        options.tensor_A.device_ref(),
        options.tensor_B.device_ref(),
        ElementComputeEpilogue(0),
        options.tensor_ref_D.device_ref());
    CUDA_CHECK(cudaDeviceSynchronize());
    options.tensor_ref_D.sync_host();
    for (int i = 0; i < options.problem_size.m(); ++i) {
      for (int j = 0; j < options.problem_size.n(); ++j) {
        options.tensor_ref_D.at({i, j}) = std::max(
          ElementD(0), 
          ElementD(options.tensor_ref_D.at({i, j}) + options.tensor_C_bias.at({0, j}))
        );
      }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    ///

    int res = run<DeviceGemmBasic>("Basic data-parallel GEMM", options);
    CUDA_CHECK(cudaDeviceSynchronize());
    options.tensor_D.sync_host();

    std::cout << (cutlass::reference::host::TensorEquals(options.tensor_D.host_view(),
                                                       options.tensor_ref_D.host_view())
                    ? "Passed"
                    : "Failed")
            << std::endl;
    return res;
}