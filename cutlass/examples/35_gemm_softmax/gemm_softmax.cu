/***************************************************************************************************
 * Copyright (c) 2017 - 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/**

*/

#include <cmath>
#include <iostream>
#include <vector>
#include <limits>

#include "cutlass/cutlass.h"
#include "cutlass/arch/memory.h"
#include "cutlass/arch/memory_sm75.h"
#include "cutlass/gemm/device/gemm_complex.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/host_tensor.h"

#include "cutlass/util/reference/host/gemm_complex.h"
#include "cutlass/util/reference/host/tensor_reduce.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/error_metrics.h"
#include "cutlass/util/tensor_view_io.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

#include "gemm_with_softmax.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

#define TRACE(x) { std::cout << "gemm_softmax.cu:" << __LINE__ << "  " << x << std::endl; }

/////////////////////////////////////////////////////////////////////////////////////////////////

enum class Disposition {
  kPassed,
  kIncorrect,
  kNotVerified
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// Command line options parsing
struct Options {

  bool help;
  cutlass::gemm::GemmCoord problem_size;
  int batch_count;
  int iterations;
  unsigned seed;
  float alpha;
  float beta;
  bool verification_enabled;
  double tolerance;

  Options():
    help(false),
    problem_size({16, 24, 64}),
    batch_count(1),             // As a temporary limitation to the test bench, batch count must be 1. The kernels support arbitrary batching.
    iterations(20),
    seed(2022),
    alpha(1),
    beta(),
    verification_enabled(true),
    tolerance(0.01)
  { }

  bool valid() {

    return true;
  }

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
    }

    cmd.get_cmd_line_argument("m", problem_size.m());
    cmd.get_cmd_line_argument("n", problem_size.n());
    cmd.get_cmd_line_argument("k", problem_size.k());

    cmd.get_cmd_line_argument("alpha", alpha);
    cmd.get_cmd_line_argument("beta", beta);

    cmd.get_cmd_line_argument("iterations", iterations);
    cmd.get_cmd_line_argument("verify", verification_enabled);
    cmd.get_cmd_line_argument("seed", seed);
    cmd.get_cmd_line_argument("tolerance", tolerance);
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "35_gemm_softmax example\n\n"
      << "  This example uses the CUTLASS Library to compute GEMM + Softmax for arbitrary problem sizes.\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement.\n\n"
      << "  --m=<int>                   GEMM M dimension\n"
      << "  --n=<int>                   GEMM N dimension\n"
      << "  --k=<int>                   GEMM K dimension\n"
      << "  --alpha=<f32>               Epilogue scalar alpha\n"
      << "  --beta=<f32>                Epilogue scalar beta\n\n"
      << "  --seed=<int>                Random number seed (1*)\n\n"
      << "  --iterations=<int>          Number of profiling iterations to perform (0 to disable profiling).\n\n"
      << "  --verify=<bool>             If true, performs reference calculation.\n\n"
      << "  --tolerance <float>         Error tolerance\n"
    ;

    out << "\n\nExamples:\n\n"
      << "$ ./examples/35_gemm_softmax/35_gemm_softmax --m=1024 --n=512 \\\n"
      << "     --alpha=2 --beta=0.707 \n\n";

    return out;
  }

  /// Returns true if the environment and Toolkit support this
  bool supported(bool verbose = true) const {

    // Ampere Tensor Core operations exposed with mma.sync and ldmatrix are first available
    // in CUDA 11.0.
    //
    // CUTLASS must be compiled with CUDA 11.0 Toolkit to run these examples.
    if (!(__CUDACC_VER_MAJOR__ >= 11)) {
      if (verbose) {
        std::cerr << "Ampere Tensor Core operations must be compiled with CUDA 11.0 Toolkit or later." << std::endl;
      }
      return false;
    }

    cudaDeviceProp props;

    cudaError_t error = cudaGetDeviceProperties(&props, 0);
    if (error != cudaSuccess) {
      if (verbose) {
        std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
      }
      return false;
    }

    if (!((props.major * 10 + props.minor) >= 80)) {
      if (verbose) {
        std::cerr << "Ampere Tensor Core operations must be run on a machine with compute capability at least 80."
                  << std::endl;
      }
      return false;
    }

    return true;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

struct Testbed {

  //
  // Type definitions
  //


  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = cutlass::half_t;
  using ElementD = cutlass::half_t;
  using ElementCompute = float;
  using ElementSoftmax = cutlass::half_t;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;

  using GemmSoftmax = cutlass::GemmSoftmax<
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC,
    ElementCompute
  >;

  using ElementN = typename GemmSoftmax::ElementN;
  using LayoutC = typename GemmSoftmax::LayoutC;

  //
  // Data members
  //

  Options const &options;

  cutlass::HostTensor<ElementA, LayoutA>        tensor_A;
  cutlass::HostTensor<ElementB, LayoutB>        tensor_B;
  cutlass::HostTensor<ElementC, LayoutC>        tensor_C;
  cutlass::HostTensor<ElementD, LayoutC>        tensor_D;
  cutlass::HostTensor<ElementN, LayoutC>        tensor_N;
  cutlass::HostTensor<ElementSoftmax, LayoutC>  tensor_Softmax;

  cutlass::HostTensor<ElementD, LayoutC>        reference_D;
  cutlass::HostTensor<ElementN, LayoutC>        reference_N;
  cutlass::HostTensor<ElementSoftmax, LayoutC>  reference_Softmax;

  //
  // Methods
  //

  Testbed(
    Options const &options_
  ):
    options(options_)
  {

    tensor_A.reset({options.problem_size.m(), options.problem_size.k()});
    tensor_B.reset({options.problem_size.k(), options.problem_size.n()});

    tensor_C.reset({options.problem_size.m(), options.problem_size.n()});
    tensor_D.reset({options.problem_size.m(), options.problem_size.n()});

    tensor_N.reset({options.problem_size.m(), 1});
    tensor_Softmax.reset({options.problem_size.m(), options.problem_size.n()});

    reference_D.reset({options.problem_size.m(), options.problem_size.n()}, false);
    reference_N.reset({options.problem_size.m(), 1}, false);
    reference_Softmax.reset({options.problem_size.m(), options.problem_size.n()}, false);
  }

  /// Run
  Disposition run() {

    Disposition disposition = Disposition::kNotVerified;

    //
    // Initialize the workspace
    //

    initialize();

    //
    // Launch device kernel
    //
    cutlass::Status status = cutlass::Status::kSuccess;

    status = execute_device_kernel();

    if (status != cutlass::Status::kSuccess) {
      std::cerr << "Device execution failed." << std::endl;
      return disposition;
    }

    cudaError_t result = cudaDeviceSynchronize();
    if (result != cudaSuccess) {
      std::cerr << "Device synchronize failed with error "
        << cudaGetErrorString(result) << std::endl;
      return disposition;
    }

    //
    // Compute the reference
    //
    compute_reference();

    //
    // Verify
    //

    if (options.verification_enabled) {

      bool passed = verify();

      if (passed) {
        disposition = Disposition::kPassed;
      }
      else {
        disposition = Disposition::kIncorrect;
      }
    }

    //
    // Profiling
    //
    if (options.iterations) {
      profile();
    }

    return disposition;
  }

  /// Random initialization
  void initialize() {

    cutlass::reference::host::TensorFillRandomUniform(
      tensor_A.host_view(),
        options.seed,
        ElementD(5),
        ElementD(-5),
        0
      );

    cutlass::reference::host::TensorFillRandomUniform(
      tensor_B.host_view(),
        options.seed + 19,
        ElementD(5),
        ElementD(-5),
        0
      );

    cutlass::reference::host::TensorFill(
      reference_D.host_view(),
      ElementD()
      );

    cutlass::reference::host::TensorFill(
      reference_N.host_view(),
      ElementN()
    );

    cutlass::reference::host::TensorFill(
      reference_Softmax.host_view(),
      ElementSoftmax()
    );

    tensor_A.sync_device();
    tensor_B.sync_device();
    tensor_D.sync_device();
    tensor_N.sync_device();
    tensor_Softmax.sync_device();
  }

  cutlass::Status execute_device_kernel() {

    cutlass::Status status = cutlass::Status::kSuccess;

    //
    // Setup arguments
    //

    GemmSoftmax::Arguments args(
      options.problem_size,
      options.batch_count,
      tensor_A.device_ref(),
      tensor_B.device_ref(),
      tensor_C.device_ref(),
      tensor_D.device_ref(),
      {
        ElementCompute(options.alpha),
        ElementCompute(options.beta)
      },
      tensor_N.device_ref(),
      tensor_Softmax.device_ref()
    );

    //
    // Launch
    //

    GemmSoftmax gemm_softmax;

    // Initialize
    status = gemm_softmax.initialize(args);
    if (status != cutlass::Status::kSuccess) {
      return status;
    }

    // Run
    status = gemm_softmax();

    return status;
  }

  /// Reference calculation
  void compute_reference() {

    // Compute GEMM

    cutlass::reference::host::GemmComplex(
      options.problem_size,
      options.alpha,
      tensor_A.host_ref(),
      cutlass::ComplexTransform::kNone,
      tensor_B.host_ref(),
      cutlass::ComplexTransform::kNone,
      options.beta,
      tensor_C.host_ref(),
      reference_D.host_ref(),
      double()
    );

    // Compute the norm
    for (int m = 0; m < options.problem_size.m(); ++m) {
      reference_N.at({m, 0}) = reference_D.at({m, 0});
      for (int n = 1; n < options.problem_size.n(); ++n) {
        reference_N.at({m, 0}) = std::max(reference_N.at({m, 0}), ElementN(reference_D.at({m, n})));
      }
    }

    // Compute softmax
    for (int m = 0; m < options.problem_size.m(); ++m) {

      float sum = float();

      for (int n = 0; n < options.problem_size.n(); ++n) {
        sum += std::exp( float(reference_D.at({m, n})) - float(reference_N.at({m, 0})) );
      }

      float inv_sum = float(1.0f / sum);

      for (int n = 0; n < options.problem_size.n(); ++n) {

        reference_Softmax.at({m, n}) = ElementSoftmax(
          std::exp( float(reference_D.at({m, n})) - float(reference_N.at({m, 0})) ) * inv_sum
        );
      }
    }
  }

  /// Emits all tensor values
  void emit_results() {
    std::cout << "D = \n" << tensor_D.host_view() << "\n\n";
    std::cout << "N = \n" << tensor_N.host_view() << "\n\n";
    std::cout << "Softmax = \n" << tensor_Softmax.host_view() << "\n\n";
    std::cout << "Reference N = \n" << reference_N.host_view() << "\n\n";
    std::cout << "Reference D = \n" << reference_D.host_view() << "\n\n";
    std::cout << "Reference Softmax = \n" << reference_Softmax.host_view() << "\n\n";
  }

  /// Verifies the reference matches
  bool verify() {

    tensor_D.sync_host();
    tensor_N.sync_host();
    tensor_Softmax.sync_host();

    double const kThreshold = options.tolerance;

    // Verification checks - set any of these to 'true' to override the verification checks.
    bool verified_D = false;
    bool verified_N = false;
    bool verified_Softmax = false;

    // Verify softmax output
    if (!verified_D) {

      double norm_diff = cutlass::reference::host::TensorNormDiff(
        tensor_D.host_view(),
        reference_D.host_view());

      double norm_reference = cutlass::reference::host::TensorNorm(
        reference_D.host_view());

      double rel_error = norm_diff / norm_reference;

      if (rel_error > kThreshold) {
        std::cerr << "\n\nTensor D Relative error: " << rel_error << std::endl;
      }
      else {
        verified_D = true;
      }
    }

    if (!verified_N) {

      double norm_diff = cutlass::reference::host::TensorNormDiff(
        tensor_N.host_view(),
        reference_N.host_view());

      double norm_reference = cutlass::reference::host::TensorNorm(
        reference_N.host_view());

      double rel_error = norm_diff / norm_reference;

      if (rel_error > kThreshold) {
        std::cerr << "\n\nTensor N Relative error: " << rel_error << std::endl;
      }
      else {
        verified_N = true;
      }
    }

    if (!verified_Softmax) {

      double norm_diff = cutlass::reference::host::TensorNormDiff(
        tensor_Softmax.host_view(),
        reference_Softmax.host_view());

      double norm_reference = cutlass::reference::host::TensorNorm(
        reference_Softmax.host_view());

      double rel_error = norm_diff / norm_reference;

      if (rel_error > kThreshold) {
        std::cerr << "\n\nSoftmax Relative error: " << rel_error << std::endl;
      }
      else {
        verified_Softmax = true;
      }
    }

    if (!verified_D || !verified_N || !verified_Softmax) {

      std::cerr << "Verification check failed for tensor Softmax" << std::endl;

      emit_results();

      // Summarize which checks failed
      if (!verified_D) {
        std::cerr << "Verification of D tensor failed\n";
      }

      if (!verified_N) {
        std::cerr << "Verification of N tensor failed\n";
      }

      if (!verified_Softmax) {
        std::cerr << "Verification of Softmax tensor failed\n";
      }

      return false;
    }

    return true;
  }

  /// Profiles
  bool profile() {

    //
    // Profile
    //

    cutlass::Status status = cutlass::Status::kSuccess;
    cudaError_t result;
    cudaEvent_t events[2];
    int const kIterations = options.iterations;

    for (cudaEvent_t &evt : events) {
      result = cudaEventCreate(&evt);
      if (result != cudaSuccess) {
        std::cerr << "cudaEventCreate failed with error " << cudaGetErrorString(result) << std::endl;
        return false;
      }
    }

    result = cudaEventRecord(events[0]);

    if (result != cudaSuccess) {
      std::cerr << "cudaEventRecord() failed with error " << cudaGetErrorString(result) << std::endl;
      return false;
    }

    for (int iter = 0; iter < kIterations; ++iter) {

      status = execute_device_kernel();

      if (status != cutlass::Status::kSuccess) {
        std::cerr << "Device execution failed." << std::endl;
        return false;
      }
    }

    result = cudaEventRecord(events[1]);

    if (result != cudaSuccess) {
      std::cerr << "cudaEventRecord() failed with error " << cudaGetErrorString(result) << std::endl;
      return false;
    }

    result = cudaDeviceSynchronize();

    if (result != cudaSuccess) {
      std::cerr << "cudaDeviceSynchronize() failed with error " << cudaGetErrorString(result) << std::endl;
      return false;
    }

    float elapsed_ms = 0;
    result = cudaEventElapsedTime(&elapsed_ms, events[0], events[1]);

    if (result != cudaSuccess) {
      std::cerr << "cudaEventElapsedTime() failed with error " << cudaGetErrorString(result) << std::endl;
      return false;
    }

    for (cudaEvent_t &evt : events) {
      result = cudaEventDestroy(evt);
      if (result != cudaSuccess) {
        std::cerr << "cudaEventDestroy() failed with error " << cudaGetErrorString(result) << std::endl;
        return false;
      }
    }

    int64_t flops = int64_t(options.problem_size.m()) * options.problem_size.n() * options.problem_size.k() * 2;
    int64_t bytes = (sizeof(ElementD) * 2 + sizeof(ElementSoftmax)) * options.problem_size.m() * options.problem_size.n();

    double gflops_per_second = double(flops) * kIterations / double(elapsed_ms / 1000.0f) / double(1.0e9);
    double gbytes_per_second = double(bytes) * kIterations / double(elapsed_ms / 1000.0f) / double(1 << 30);

    std::cout << "         Problem: "
              << options.problem_size.m() << "-by-" << options.problem_size.n() << "-by-" << options.problem_size.k()
              << std::endl;

    std::cout << "         Runtime: " << elapsed_ms << " ms\n" << std::endl;

    std::cout << "          GFLOPs: " << gflops_per_second << "  GFLOPs" << std::endl;
    std::cout << "Memory bandwidth: " << gbytes_per_second << "  GiB/s" << std::endl;

    return true;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, const char **argv) {

  // Options parsing
  Options options;
  options.parse(argc, argv);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  if (!options.supported()) {
    return 0;
  }

  // Run
  Testbed testbed(options);

  Disposition disposition = testbed.run();

  std::cout << std::endl;

  switch (disposition) {
    case Disposition::kPassed:
      std::cout << "Passed" << std::endl;
      break;
    case Disposition::kIncorrect:
      std::cout << "Incorrect" << std::endl;
      break;
    case Disposition::kNotVerified:
      std::cout << "Not verified" << std::endl;
      break;
  }

  return (disposition == Disposition::kPassed ? 0 : -1);
}


/////////////////////////////////////////////////////////////////////////////////////////////////

