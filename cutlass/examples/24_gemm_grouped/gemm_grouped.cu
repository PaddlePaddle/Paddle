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

/*! \file
    \brief GEMM Grouped Example.

    This workload computes a batch of GEMM operations with distinct problem sizes. Pointers to matrices
    in Global Memory are passed to the kernel in array (also held in Global Memory). Similarly,
    leading dimensions and problem sizes are stored in arrays in GMEM.

    This differs from "Batched Array" GEMM because the size of each GEMM problem in the Grouped GEMM
    concept may be distinct. 

    This benchmark program initializes a workspace with random problem sizes for a given number of
    groups. Command line options enable overriding M, N, and/or K dimensions with uniform values to
    model problems more similar to the traditional batched GEMM.

    Additionally, problem sizes are collected and binned to compute the same problem as a series of
    conventional batched GEMMs (setup for this problem is not timed). This demonstrates the performance
    enhancement achieved by implementing a specialized grouped GEMM kernel.

    Examples:

      # Runs a grouped GEMM with 100 random problem sizes
      $ ./examples/24_gemm_grouped/24_gemm_grouped --groups=100

      # Runs a grouped GEMM with 100 random problem sizes (with GEMM-K dimension equal to 1024)
      $ ./examples/24_gemm_grouped/24_gemm_grouped --groups=100 --k=1024 --verbose=true

      # Runs a grouped GEMM that is equivalent to a batched GEMM
      $ ./examples/24_gemm_grouped/24_gemm_grouped --groups=100 --m=2048 --n=1024 --k=1024 --verbose=true

      # Execute Grouped GEMM and profile with NSight
      $ nv-nsight-cu-cli ./examples/24_gemm_grouped/24_gemm_grouped --m=256 --n=256 --k=256 --verbose=true \
                                                                    --iterations=1 --reference-check=false

*/

/////////////////////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <unordered_map>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/device/gemm_universal.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/gemm_complex.h"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_norm.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Result structure
struct Result {

  double runtime_ms;
  double gflops;
  cutlass::Status status;
  cudaError_t error;
  bool passed;

  //
  // Methods
  //

  Result(
    double runtime_ms = 0,
    double gflops = 0,
    cutlass::Status status = cutlass::Status::kSuccess,
    cudaError_t error = cudaSuccess
  ):
    runtime_ms(runtime_ms), gflops(gflops), status(status), error(error), passed(true) { }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Hash function for cutlass::gemm::GemmCoord
struct HashGemmCoord {
  size_t operator()(cutlass::gemm::GemmCoord const &problem) const {
    std::hash<int> hasher;
    return (hasher(problem.m() * 3)) ^ (hasher(1 + problem.n() * 5)) ^ (hasher(2 + problem.k() * 7));
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// Command line options parsing
struct Options {

  bool help;
  bool error;
  bool reference_check;

  std::vector<cutlass::gemm::GemmCoord> problem_sizes;

  // problem size bins
  std::unordered_map<
    cutlass::gemm::GemmCoord,
    std::vector<int32_t>,
    HashGemmCoord> problem_bins;

  int alignment;
  int problem_count;
  int iterations;
  int cuda_streams;
  bool verbose;
  float alpha;
  float beta;
  std::string benchmark_path;

  std::string   output_tag;
  std::ofstream output_file;

  //
  // Methods
  // 

  Options():
    help(false),
    error(false),
    alignment(8),
    reference_check(true),
    problem_count(15),
    iterations(20),
    cuda_streams(0),
    verbose(false),
    alpha(1),
    beta()
  { }

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    cmd.get_cmd_line_argument("alignment", alignment, 8);
    cmd.get_cmd_line_argument("groups", problem_count, 15);
    cmd.get_cmd_line_argument("alpha", alpha, 1.0f);
    cmd.get_cmd_line_argument("beta", beta, 0.0f);    
    cmd.get_cmd_line_argument("iterations", iterations, 20);
    cmd.get_cmd_line_argument("streams", cuda_streams, 0);
    cmd.get_cmd_line_argument("verbose", verbose, false);
    cmd.get_cmd_line_argument("reference-check", reference_check, true);
    cmd.get_cmd_line_argument("benchmark", benchmark_path);

    std::string output_path;
    cmd.get_cmd_line_argument("tag", output_tag);
    cmd.get_cmd_line_argument("output_file", output_path);

    if (!output_path.empty()) {

      std::ios_base::openmode open_mode = std::ios_base::out;

      std::ifstream input_file(output_path.c_str());

      if (input_file.good()) {
        open_mode = std::ios_base::app;
        input_file.close();
      }

      output_file.open(output_path.c_str(), open_mode);

      if (output_file.good() && open_mode != std::ios_base::app) {
        output_file << "Tag,Provider,Kind,Groups,Runtime,GFLOPs\n";
      }
    }

    // Decide how to initialize the problems
    if (!benchmark_path.empty()) {
      if (!benchmark_problems()) {
        error = true;
        problem_sizes.clear();
        return;
      }
    }
    else {
      randomize_problems(cmd);
    }

    // Post-process the problem sizes
    bin_problems();
  }

  void randomize_problems(cutlass::CommandLine &cmd) {

    //
    // For now, randomly choose the problem sizes.
    //

    int cmd_line_m = -1;
    int cmd_line_n = -1;
    int cmd_line_k = -1;

    cmd.get_cmd_line_argument("m", cmd_line_m);
    cmd.get_cmd_line_argument("n", cmd_line_n);
    cmd.get_cmd_line_argument("k", cmd_line_k);

    problem_sizes.reserve(problem_count);

    for (int i = 0; i < problem_count; ++i) {

      int m = cmd_line_m;
      int n = cmd_line_n;
      int k = cmd_line_k;

      if (m < 1) {
        m = alignment * ((rand() % 256) + 1);
      }

      if (n < 1) {
        n = alignment * ((rand() % 256) + 1);
      }

      if (k < 1) {
        k = alignment * ((rand() % 256) + 1);
      }

      cutlass::gemm::GemmCoord problem(m, n, k);

      problem_sizes.push_back(problem);
    }
  }

  /// Load a benchmark
  bool benchmark_problems() {
    std::ifstream file(benchmark_path);
    if (!file.good()) {
      return false;
    }

    while (file.good()) {

      int idx = -1;
      std::string extent_str;

      file >> idx >> extent_str;

      if (idx < 0 || extent_str.empty()) {
        break;
      }

      cutlass::gemm::GemmCoord extent;
      std::vector<std::string> tokens;

      cutlass::CommandLine::tokenize(tokens, extent_str, 'x');

      for (int i = 0; i < int(tokens.size()); ++i) {
        int x = std::atoi(tokens.at(i).c_str());

        // round up
        if (x % alignment) {
          x += (alignment - (x % alignment));
        }

        extent.at(i) = x;
      }

      if (extent.product()) {
        problem_sizes.push_back(extent);
      }
    }

    return true;
  }

  /// Post processes the problems
  void bin_problems() {

    problem_count = int(problem_sizes.size());

    //
    // Insert the problem sizes into a sorted container class. This is *NOT* necessary
    // to run the CUTLASS kernel, but it enables the execution of cublas's batched GEMM.
    //
    for (int i = 0; i < int(problem_sizes.size()); ++i) {
      auto it = problem_bins.find(problem_sizes.at(i));
      if (it == problem_bins.end()) {
        problem_bins.insert({problem_sizes.at(i), std::vector<int32_t>({i}) });
      }
      else {
        it->second.push_back(i);
      }
    }
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "24_gemm_grouped\n\n"
      << "  This example profiles the performance of a 'grouped' GEMM kernel. This is similar to batched GEMM\n"
      << "  in that multiple, independent GEMMs are computed by one grid launch. It differs in that each\n"
      << "  'group' may compute a unique problem size. Problem sizes and pointers to matrices are both stored\n"
      << "  in device Global Memory and loaded by the kernel.\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement.\n\n"
      << "  --benchmark=<str>           Executes a benchmark problem size.\n"
      << "  --output_file=<str>         Path to a CSV file to output results. If it exists already, results are appended.\n"
      << "  --tag=<str>                 String tag to prepend to the CSV file.\n"
      << "  --groups=<int>              Number of individual GEMM problems (default: --groups=15)\n"
      << "  --m=<int>                   Sets the M dimension for all groups. Otherwise, it is selected randomly\n"
      << "  --n=<int>                   Sets the N dimension for all groups. Otherwise, it is selected randomly\n"
      << "  --k=<int>                   Sets the K dimension for all groups. Otherwise, it is selected randomly\n"
      << "  --alpha=<f32>               Epilogue scalar alpha (real part)\n"
      << "  --beta=<f32>                Epilogue scalar beta (real part)\n\n"
      << "  --iterations=<int>          Number of profiling iterations to perform.\n"
      << "  --reference-check=<bool>    If true, performs reference check.\n"
      << "  --verbose=<bool>            If true, prints problem sizes and batching structure.\n";

    out << "\n\nExamples:\n\n"

      << "# Runs a grouped GEMM with 100 random problem sizes\n"
      << "$ ./examples/24_gemm_grouped/24_gemm_grouped --groups=100\n\n"

      << "# Runs a grouped GEMM with 100 random problem sizes (with GEMM-K dimension equal to 1024)\n"
      << "$ ./examples/24_gemm_grouped/24_gemm_grouped --groups=100 --k=1024 --verbose=true\n\n"

      << "# Runs a grouped GEMM that is equivalent to a batched GEMM\n"
      << "$ ./examples/24_gemm_grouped/24_gemm_grouped --groups=100 --m=2048 --n=1024 --k=1024 --verbose=true\n\n"

      << "# Runs a grouped GEMM problem given an externally supplied benchmark file. This is a text file in which\n"
      << "# Each line contains a unique group index and an MxNxK triple indicating problemsize.\n"
      << "#\n"
      << "# For example, assume the following are the contents of 'problems.txt'\n"
      << "#\n"
      << "# 0 1024x256x520\n"
      << "# 1 520x264x1024\n"
      << "# 2 96x48x1024\n"
      << "#\n"
      << "$ ./examples/24_gemm_grouped/24_gemm_grouped --benchmark=problems.txt\n\n"

      << "# Execute Grouped GEMM and profile with NSight\n"
      << "$ nv-nsight-cu-cli ./examples/24_gemm_grouped/24_gemm_grouped --m=256 --n=256 --k=256 --verbose=true --iterations=1 --reference-check=false\n\n";

    return out;
  }

  /// Compute performance in GFLOP/s
  double gflops(double runtime_s) const {

    // Number of real-valued multiply-adds 
    int64_t fmas = int64_t();

    for (auto const & problem : problem_sizes) {
      fmas += problem.product();
    }
    
    // Two flops per multiply-add
    return 2.0 * double(fmas) / double(1.0e9) / runtime_s;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Gemm, typename GemmBatched>
class TestbedGrouped {
public:

  //
  // Type definitions
  //

  using ElementA = typename Gemm::ElementA;
  using ElementB = typename Gemm::ElementB;
  using ElementC = typename Gemm::ElementC;
  using ElementAccumulator = typename Gemm::ElementAccumulator;

  using EpilogueOutputOp = typename Gemm::GemmKernel::Epilogue::OutputOp;
  using ElementCompute = typename EpilogueOutputOp::ElementCompute;

  using LayoutA = typename Gemm::LayoutA;
  using LayoutB = typename Gemm::LayoutB;
  using LayoutC = typename Gemm::LayoutC;

  using MatrixCoord = typename LayoutC::TensorCoord;

private:

  //
  // Data members
  //

  Options & options;

  /// Initialization
  cutlass::Distribution::Kind init_A;
  cutlass::Distribution::Kind init_B;
  cutlass::Distribution::Kind init_C;
  uint32_t seed;

  cutlass::DeviceAllocation<cutlass::gemm::GemmCoord> problem_sizes_device;

  std::vector<int64_t> offset_A;
  std::vector<int64_t> offset_B;
  std::vector<int64_t> offset_C;
  std::vector<int64_t> offset_D;

  std::vector<int64_t> lda_host;
  std::vector<int64_t> ldb_host;
  std::vector<int64_t> ldc_host;
  std::vector<int64_t> ldd_host;

  cutlass::DeviceAllocation<int64_t> lda;
  cutlass::DeviceAllocation<int64_t> ldb;
  cutlass::DeviceAllocation<int64_t> ldc;
  cutlass::DeviceAllocation<int64_t> ldd;

  cutlass::DeviceAllocation<ElementA> block_A;
  cutlass::DeviceAllocation<ElementB> block_B;
  cutlass::DeviceAllocation<ElementC> block_C;
  cutlass::DeviceAllocation<ElementC> block_D;

  cutlass::DeviceAllocation<ElementA *> ptr_A;
  cutlass::DeviceAllocation<ElementB *> ptr_B;
  cutlass::DeviceAllocation<ElementC *> ptr_C;
  cutlass::DeviceAllocation<ElementC *> ptr_D;

public:

  //
  // Methods
  //

  TestbedGrouped(
    Options &options_,
    cutlass::Distribution::Kind init_A_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_B_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_C_ = cutlass::Distribution::Uniform,
    uint32_t seed_ = 3080
  ):
    options(options_), init_A(init_A_), init_B(init_B_), init_C(init_C_), seed(seed_) { }

  int problem_count() const {
    return options.problem_count;
  }

private:

  /// Helper to initialize a tensor view
  template <typename Element>
  void initialize_tensor_(
    Element *ptr,
    size_t capacity, 
    cutlass::Distribution::Kind dist_kind,
    uint32_t seed) {

    if (dist_kind == cutlass::Distribution::Uniform) {

      Element scope_max, scope_min;
      int bits_input = cutlass::sizeof_bits<Element>::value;
      int bits_output = cutlass::sizeof_bits<typename Gemm::ElementC>::value;

      if (bits_input == 1) {
        scope_max = 2;
        scope_min = 0;
      } else if (bits_input <= 8) {
        scope_max = 2;
        scope_min = -2;
      } else if (bits_output == 16) {
        if (cutlass::sizeof_bits<ElementAccumulator>::value <= 16) {
          scope_max = 5;
          scope_min = -5;
        }
        else {
          scope_max = 8;
          scope_min = -8;
        }
      } else {
        scope_max = 8;
        scope_min = -8;
      }

      cutlass::reference::device::BlockFillRandomUniform(
        ptr, capacity, seed, scope_max, scope_min, 0);
    } 
    else if (dist_kind == cutlass::Distribution::Gaussian) {

      cutlass::reference::device::BlockFillRandomGaussian(
        ptr, capacity, seed, Element(), Element(0.5f));
    }
    else if (dist_kind == cutlass::Distribution::Sequential) {

      // Fill with increasing elements
      cutlass::reference::device::BlockFillSequential(
        ptr, capacity, Element(1), Element());
    } 
    else {

      // Fill with all 1s
      cutlass::reference::device::BlockFillSequential(
        ptr, capacity, Element(), Element(1));
    }
  }

  /// Verbose printing of problem sizes
  void print_problem_sizes_() {

    // Print groups
    std::cout << problem_count() << " groups:\n";  
    
    int32_t idx = 0;
    int64_t total_tiles = 0;

    for (auto const & problem : options.problem_sizes) {

      int tiles = 
        ((problem.m() + Gemm::ThreadblockShape::kM - 1) / Gemm::ThreadblockShape::kM) * 
        ((problem.n() + Gemm::ThreadblockShape::kN - 1) / Gemm::ThreadblockShape::kN);

      total_tiles += tiles;

      std::cout << "  [" << idx << "]: " 
        << problem.m() << "-by-" << problem.n() << "-by-" << problem.k() 
        << " (" << tiles << " threadblock tiles)" << "\n";

      ++idx;
    }

    // Print batched GEMM equivalent
    size_t bin_idx = 0;
    size_t problem_count_check = 0;
    std::cout << "\nConventionally executed as " << options.problem_bins.size() << " batched GEMMs:\n";
    for (auto const & bin : options.problem_bins) {

      std::cout << "  [" << bin_idx << "]: " 
        << bin.first.m() << "-by-" << bin.first.n() << "-by-" << bin.first.k() 
        << ", batch count: " << bin.second.size() << "\n";

      ++bin_idx;
      problem_count_check += bin.second.size();
    }

    if (problem_count_check != problem_count()) {
      std::cout << "\n***\nERROR in BINNING LOGIC!\n***\n" << std::endl;
    }
  }

  /// Initializes data structures
  void initialize_() {

    //
    // Choose random problem sizes
    //

    // construct a few problems of random sizes
    srand(seed);

    int64_t total_elements_A = 0;
    int64_t total_elements_B = 0;
    int64_t total_elements_C = 0;
    int64_t total_elements_D = 0;


    lda_host.resize(problem_count());
    ldb_host.resize(problem_count());
    ldc_host.resize(problem_count());
    ldd_host.resize(problem_count());

    for (int32_t i = 0; i < problem_count(); ++i) {

      auto problem = options.problem_sizes.at(i);

      lda_host.at(i) = LayoutA::packed({problem.m(), problem.k()}).stride(0);
      ldb_host.at(i) = LayoutB::packed({problem.k(), problem.n()}).stride(0);
      ldc_host.at(i) = LayoutC::packed({problem.m(), problem.n()}).stride(0);
      ldd_host.at(i) = LayoutC::packed({problem.m(), problem.n()}).stride(0);

      offset_A.push_back(total_elements_A);
      offset_B.push_back(total_elements_B);
      offset_C.push_back(total_elements_C);
      offset_D.push_back(total_elements_D);

      int64_t elements_A = problem.m() * problem.k();
      int64_t elements_B = problem.k() * problem.n();
      int64_t elements_C = problem.m() * problem.n();
      int64_t elements_D = problem.m() * problem.n();

      total_elements_A += elements_A;
      total_elements_B += elements_B;
      total_elements_C += elements_C;
      total_elements_D += elements_D;
    }

    problem_sizes_device.reset(problem_count());
    problem_sizes_device.copy_from_host(options.problem_sizes.data());

    lda.reset(problem_count());
    ldb.reset(problem_count());
    ldc.reset(problem_count());
    ldd.reset(problem_count());

    lda.copy_from_host(lda_host.data());
    ldb.copy_from_host(ldb_host.data());
    ldc.copy_from_host(ldc_host.data());
    ldd.copy_from_host(ldd_host.data());

    //
    // Assign pointers
    //

    block_A.reset(total_elements_A);
    block_B.reset(total_elements_B);
    block_C.reset(total_elements_C);
    block_D.reset(total_elements_D);

    std::vector<ElementA *> ptr_A_host(problem_count());
    std::vector<ElementB *> ptr_B_host(problem_count());
    std::vector<ElementC *> ptr_C_host(problem_count());
    std::vector<ElementC *> ptr_D_host(problem_count());

    for (int32_t i = 0; i < problem_count(); ++i) {
      ptr_A_host.at(i) = block_A.get() + offset_A.at(i);
      ptr_B_host.at(i) = block_B.get() + offset_B.at(i);
      ptr_C_host.at(i) = block_C.get() + offset_C.at(i);
      ptr_D_host.at(i) = block_D.get() + offset_D.at(i);
    }

    ptr_A.reset(problem_count());
    ptr_A.copy_from_host(ptr_A_host.data());
    
    ptr_B.reset(problem_count());
    ptr_B.copy_from_host(ptr_B_host.data());
    
    ptr_C.reset(problem_count());
    ptr_C.copy_from_host(ptr_C_host.data());
    
    ptr_D.reset(problem_count());
    ptr_D.copy_from_host(ptr_D_host.data());

    //
    // Initialize the problems of the workspace
    //

    initialize_tensor_(block_A.get(), total_elements_A, init_A, seed * 2021);
    initialize_tensor_(block_B.get(), total_elements_B, init_B, seed * 2022);
    initialize_tensor_(block_C.get(), total_elements_C, init_C, seed * 2023);

    cutlass::reference::device::BlockFillSequential(
      block_D.get(), total_elements_D, ElementC(), ElementC());
  }

  /// Verifies the result is a GEMM
  bool verify_() {

    bool passed = true;

    for (int32_t i = 0; i < problem_count(); ++i) {
      cutlass::gemm::GemmCoord problem = options.problem_sizes.at(i);

      LayoutA layout_A(lda_host.at(i));
      LayoutB layout_B(ldb_host.at(i));
      LayoutC layout_C(ldc_host.at(i));
      LayoutC layout_D(ldd_host.at(i));

      MatrixCoord extent_A{problem.m(), problem.k()};
      MatrixCoord extent_B{problem.k(), problem.n()};
      MatrixCoord extent_C{problem.m(), problem.n()};
      
      cutlass::TensorView<ElementA, LayoutA> view_A(block_A.get() + offset_A.at(i), layout_A, extent_A);
      cutlass::TensorView<ElementB, LayoutB> view_B(block_B.get() + offset_B.at(i), layout_B, extent_B);
      cutlass::TensorView<ElementC, LayoutC> view_C(block_C.get() + offset_C.at(i), layout_C, extent_C);

      cutlass::DeviceAllocation<ElementC>    block_Ref(layout_D.capacity(extent_C));
      cutlass::TensorView<ElementC, LayoutC> view_Ref_device(block_Ref.get(), layout_D, extent_C);

      // Reference GEMM
      cutlass::reference::device::GemmComplex<
          ElementA, LayoutA,
          ElementB, LayoutB,
          ElementC, LayoutC, 
          ElementCompute, ElementAccumulator
      >(
        problem,
        options.alpha, 
        view_A,
        Gemm::kTransformA,
        view_B,
        Gemm::kTransformB,
        options.beta, 
        view_C, 
        view_Ref_device, 
        ElementAccumulator(0)
      );

      // Copy to host memory
      std::vector<ElementC> matrix_D(layout_D.capacity(extent_C));
      std::vector<ElementC> matrix_Ref(layout_D.capacity(extent_C));

      cutlass::device_memory::copy_to_host(matrix_D.data(),   block_D.get() + offset_D.at(i), matrix_D.size());
      cutlass::device_memory::copy_to_host(matrix_Ref.data(), block_Ref.get(),                matrix_D.size());

      cutlass::TensorView<ElementC, LayoutC> view_D(  matrix_D.data(),   layout_D, extent_C);
      cutlass::TensorView<ElementC, LayoutC> view_Ref(matrix_Ref.data(), layout_D, extent_C);
      
      // Reference check
      passed = cutlass::reference::host::TensorEquals(view_D, view_Ref);

      if (!passed) {
        std::cerr << "\n***\nError - problem " << i << " failed the QA check\n***\n" << std::endl;
        return passed;
      }
    }

    return passed;
  }

public:

  /// Returns the number of threadblocks to launch if the kernel can run on the target
  /// device. Otherwise, returns zero.
  int sufficient() const {
    cudaDeviceProp properties;
    int device_idx;
    cudaError_t result = cudaGetDevice(&device_idx);

    if (result != cudaSuccess) {
      throw std::runtime_error("cudaGetDevice() API call failed.");
    }

    result = cudaGetDeviceProperties(&properties, device_idx);

    if (result != cudaSuccess) {
      throw std::runtime_error("cudaGetDeviceProperties() failed");
    }

    int occupancy = Gemm::maximum_active_blocks();

    return properties.multiProcessorCount * occupancy;

  }


  /// Executes a Grouped GEMM kernel and measures runtime.
  Result profile_grouped() {

    Result result;

    int threadblock_count = sufficient();

    // Early exit
    if (!threadblock_count) {
      std::cout << "Active CUDA device lacks hardware resources to run CUTLASS Grouped GEMM kernel." << std::endl;
      return result;
    }

    if (options.verbose) {
      print_problem_sizes_();
    }

    result.passed = false;

    // Initialize the problem
    initialize_();

    // Configure the GEMM arguments
    typename EpilogueOutputOp::Params epilogue_op(options.alpha, options.beta);

    // Configure GEMM arguments
    typename Gemm::Arguments args(
      problem_sizes_device.get(),
      problem_count(),
      threadblock_count,
      epilogue_op,
      ptr_A.get(),
      ptr_B.get(),
      ptr_C.get(),
      ptr_D.get(),
      lda.get(),
      ldb.get(),
      ldc.get(),
      ldd.get()
    );

    // Initialize the GEMM object
    Gemm gemm;

    result.status = gemm.initialize(args);

    if (result.status != cutlass::Status::kSuccess) {
      std::cerr << "Failed to initialize CUTLASS Grouped GEMM kernel." << std::endl;
      return result;
    }

    // Run the grouped GEMM object
    result.status = gemm.run();

    if (result.status != cutlass::Status::kSuccess) {
      std::cerr << "Failed to run CUTLASS Grouped GEMM kernel." << std::endl;
      return result;
    }

    // Wait for completion
    result.error = cudaDeviceSynchronize();

    if (result.error != cudaSuccess)  {
      std::cerr << "Kernel execution error: " << cudaGetErrorString(result.error);
      return result;
    }

    //
    // Verify correctness
    //
    result.passed = true;

    if (options.reference_check) {
      result.passed = verify_();
    }

    //
    // Warm-up run of the grouped GEMM object
    //
    result.status = gemm.run();

    if (result.status != cutlass::Status::kSuccess) {
      std::cerr << "Failed to run CUTLASS Grouped GEMM kernel." << std::endl;
      return result;
    }

    //
    // Construct events
    //

    cudaEvent_t events[2];

    for (auto & event : events) {
      result.error = cudaEventCreate(&event);
      if (result.error != cudaSuccess) {
        std::cerr << "cudaEventCreate() failed: " << cudaGetErrorString(result.error) << std::endl;
        return -1;
      }
    }

    // Record an event at the start of a series of GEMM operations
    result.error = cudaEventRecord(events[0]);
    if (result.error != cudaSuccess) {
      std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(result.error) << std::endl;
      return result;
    }

    //
    // Run profiling loop
    //

    for (int iter = 0; iter < options.iterations; ++iter) {
      gemm();
    }

    //
    // Stop profiling loop
    //

    // Record an event when the GEMM operations have been launched.
    result.error = cudaEventRecord(events[1]);
    if (result.error != cudaSuccess) {
      std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(result.error) << std::endl;
      return result;
    }

    // Wait for work on the device to complete.
    result.error = cudaEventSynchronize(events[1]);
    if (result.error != cudaSuccess) {
      std::cerr << "cudaEventSynchronize() failed: " << cudaGetErrorString(result.error) << std::endl;
      return result;
    }

    // Measure elapsed runtime
    float runtime_ms = 0;
    result.error = cudaEventElapsedTime(&runtime_ms, events[0], events[1]);
    if (result.error != cudaSuccess) {
      std::cerr << "cudaEventElapsed() failed: " << cudaGetErrorString(result.error) << std::endl;
      return result;
    }

    // Compute average runtime and GFLOPs.
    result.runtime_ms = double(runtime_ms) / double(options.iterations);
    result.gflops = options.gflops(result.runtime_ms / 1000.0);

    //
    // Cleanup
    //

    for (auto event : events) {
      (void)cudaEventDestroy(event);
    }
    
    int32_t idx = 0;
    int64_t total_tiles = 0;

    for (auto const & problem : options.problem_sizes) {

      int tiles = 
        ((problem.m() + Gemm::ThreadblockShape::kM - 1) / Gemm::ThreadblockShape::kM) * 
        ((problem.n() + Gemm::ThreadblockShape::kN - 1) / Gemm::ThreadblockShape::kN);

      total_tiles += tiles;
      ++idx;
    }

    std::cout << std::endl;
    std::cout << "Grouped GEMM (CUTLASS):\n"
      << "====================================================" << std::endl;

    std::cout << "    " << total_tiles << " total threadblock tiles." << std::endl;

    std::cout << std::endl;
    std::cout << "    " << "Grouped Runtime: " << result.runtime_ms << " ms" << std::endl;
    std::cout << "    " << "Grouped  GFLOPs: " << result.gflops << std::endl;

    if (options.output_file.good()) {
      options.output_file << options.output_tag << ",CUTLASS,grouped,"
        << problem_count() << "," << result.runtime_ms << "," << result.gflops << std::endl;
    }

    return result;
  }

  /// Executes a conventional batched GEMM kernel.
  Result profile_batched() {

    Result result;
    result.passed = false;

    //
    // Prepare batched GEMM environment
    //

    int32_t effective_streams = (options.cuda_streams ? options.cuda_streams : 1);

    // Array of leading dimensions used by batched GEMM calls
    std::vector<cutlass::gemm::GemmCoord> bin_problem_sizes;
    std::vector<int32_t>                  bin_count;
    std::vector<int32_t>                  bin_ldm_A;
    std::vector<int32_t>                  bin_ldm_B;
    std::vector<int32_t>                  bin_ldm_C;
    std::vector<int32_t>                  bin_start;

    std::vector<void const *> ptr_A_batched_host;
    std::vector<void const *> ptr_B_batched_host;
    std::vector<void       *> ptr_C_batched_host;

    for (auto const & bin : options.problem_bins) {
      int first_idx = bin.second.front();
      
      bin_problem_sizes.push_back(options.problem_sizes.at(first_idx));
      bin_count.push_back(int32_t(bin.second.size()));

      bin_ldm_A.push_back(static_cast<int32_t>(lda_host.at(first_idx)));
      bin_ldm_B.push_back(static_cast<int32_t>(ldb_host.at(first_idx)));
      bin_ldm_C.push_back(static_cast<int32_t>(ldc_host.at(first_idx)));

      if (ptr_A_batched_host.size() % 2) {
        ptr_A_batched_host.push_back(nullptr);
        ptr_B_batched_host.push_back(nullptr);
        ptr_C_batched_host.push_back(nullptr);
      }

      bin_start.push_back(int32_t(ptr_A_batched_host.size()));

      for (int idx : bin.second) {

        if (bin_problem_sizes.back() != options.problem_sizes.at(idx)) {
          std::cerr << "Error - failed to group problems.\n";
          return result;
        }

        if (bin_ldm_A.back() != lda_host.at(idx)) {
          std::cerr << "Error - failed to group problems.\n";
          return result;
        }

        if (bin_ldm_B.back() != ldb_host.at(idx)) {
          std::cerr << "Error - failed to group problems.\n";
          return result;
        }

        if (bin_ldm_C.back() != ldc_host.at(idx)) {
          std::cerr << "Error - failed to group problems.\n";
          return result;
        }

        ptr_A_batched_host.push_back(block_A.get() + offset_A.at(idx));
        ptr_B_batched_host.push_back(block_B.get() + offset_B.at(idx));
        ptr_C_batched_host.push_back(block_D.get() + offset_C.at(idx));
      }
    }

    // Array of GMEM pointers used by batched array GEMM calls
    cutlass::DeviceAllocation<void const *> ptr_A_batched;
    cutlass::DeviceAllocation<void const *> ptr_B_batched;
    cutlass::DeviceAllocation<void       *> ptr_C_batched;

    ptr_A_batched.reset(ptr_A_batched_host.size());
    ptr_B_batched.reset(ptr_A_batched_host.size());
    ptr_C_batched.reset(ptr_A_batched_host.size());

    ptr_A_batched.copy_from_host(ptr_A_batched_host.data());
    ptr_B_batched.copy_from_host(ptr_B_batched_host.data());
    ptr_C_batched.copy_from_host(ptr_C_batched_host.data());

    //
    // Create CUDA streams to maximize concurrency of batched-array GEMM kernels
    //
    std::vector<cudaStream_t>   cuda_streams;
    char const *provider = "CUTLASS";

    //
    // Warmup run
    //


    if (options.cuda_streams) {
      for (int i = 0; i < options.cuda_streams; ++i) {
        cudaStream_t stream;

        result.error = cudaStreamCreate(&stream);
        if (result.error != cudaSuccess) {
        std::cerr << "Failed to create CUDA stream." << std::endl;
          return result;
        }
        cuda_streams.push_back(stream);

      }
    }
    else {
      cuda_streams.push_back(nullptr);

    }

    // Use 'D' for the in/out workspace
    block_D.copy_from_device(block_C.get());

    for (int bin_idx = 0; bin_idx < int32_t(bin_problem_sizes.size()); ++bin_idx) {

      cutlass::gemm::GemmCoord const & problem = bin_problem_sizes[bin_idx];
      int32_t batch_count = bin_count[bin_idx];
      int32_t bin_start_idx = bin_start[bin_idx];
      int32_t lda = bin_ldm_A[bin_idx];
      int32_t ldb = bin_ldm_B[bin_idx];
      int32_t ldc = bin_ldm_C[bin_idx];

      void const ** ptr_A_array = ptr_A_batched.get() + bin_start[bin_idx];
      void const ** ptr_B_array = ptr_B_batched.get() + bin_start[bin_idx];
      void       ** ptr_C_array = ptr_C_batched.get() + bin_start[bin_idx];

      //
      // Initialize the CUTLASS GEMM operator
      //

      // Configure the GEMM arguments
      typename EpilogueOutputOp::Params epilogue_op(options.alpha, options.beta);

      typename GemmBatched::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kArray,
        problem,
        batch_count,
        epilogue_op,
        (void const *)ptr_A_array,
        (void const *)ptr_B_array,
        (void const *)ptr_C_array,
        (void       *)ptr_C_array,
        int64_t(),
        int64_t(),
        int64_t(),
        int64_t(),
        int64_t(lda),
        int64_t(ldb),
        int64_t(ldc),
        int64_t(ldc)
      };

      GemmBatched gemm_op;

      cutlass::Status status = gemm_op.initialize(arguments);

      if (status != cutlass::Status::kSuccess) {
        std::cerr << "CUTLASS error on line " << __LINE__ << std::endl;
        return result;
      }

      status = gemm_op();

      if (status != cutlass::Status::kSuccess) {
        std::cerr << "CUTLASS error on line " << __LINE__ << std::endl;
        return result;
      }
      
    }

    //
    // Wait for completion
    //

    result.error = cudaDeviceSynchronize();

    if (result.error != cudaSuccess)  {
      std::cerr << "Kernel execution error: " << cudaGetErrorString(result.error);
      return result;
    }

    //
    // Construct events
    //

    cudaEvent_t events[2];

    for (auto & event : events) {
      result.error = cudaEventCreate(&event);
      if (result.error != cudaSuccess) {
        std::cerr << "cudaEventCreate() failed: " << cudaGetErrorString(result.error) << std::endl;
        return -1;
      }
    }

    //
    // Wait for completion
    //

    result.error = cudaDeviceSynchronize();

    if (result.error != cudaSuccess)  {
      std::cerr << "Kernel execution error: " << cudaGetErrorString(result.error);
      return result;
    }

    // Record an event at the start of a series of GEMM operations
    result.error = cudaEventRecord(events[0]);
    if (result.error != cudaSuccess) {
      std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(result.error) << std::endl;
      return result;
    }

    //
    // Run profiling loop
    //

    int last_stream_idx = 0;

    for (int iter = 0; iter < options.iterations; ++iter) {
      
      for (int bin_idx = 0; bin_idx < int32_t(bin_problem_sizes.size()); ++bin_idx) {

        cutlass::gemm::GemmCoord const & problem = bin_problem_sizes[bin_idx];
        int32_t batch_count = bin_count[bin_idx];
        int32_t bin_start_idx = bin_start[bin_idx];
        int32_t lda = bin_ldm_A[bin_idx];
        int32_t ldb = bin_ldm_B[bin_idx];
        int32_t ldc = bin_ldm_C[bin_idx];

        void const ** ptr_A_array = ptr_A_batched.get() + bin_start[bin_idx];
        void const ** ptr_B_array = ptr_B_batched.get() + bin_start[bin_idx];
        void       ** ptr_C_array = ptr_C_batched.get() + bin_start[bin_idx];

        last_stream_idx = (bin_idx % effective_streams);

        //
        // Initialize the CUTLASS GEMM operator
        //

        // Configure the GEMM arguments
        typename EpilogueOutputOp::Params epilogue_op(options.alpha, options.beta);

        typename GemmBatched::Arguments arguments{
          cutlass::gemm::GemmUniversalMode::kArray,
          problem,
          batch_count,
          epilogue_op,
          (void const *)ptr_A_array,
          (void const *)ptr_B_array,
          (void const *)ptr_C_array,
          (void       *)ptr_C_array,
          int64_t(),
          int64_t(),
          int64_t(),
          int64_t(),
          int64_t(lda),
          int64_t(ldb),
          int64_t(ldc),
          int64_t(ldc)
        };

        GemmBatched gemm_op;

        cutlass::Status status = gemm_op.initialize(arguments);

        if (status != cutlass::Status::kSuccess) {
          std::cerr << "CUTLASS error on line " << __LINE__ << std::endl;
          return result;
        }

        status = gemm_op(cuda_streams[last_stream_idx]);

        if (status != cutlass::Status::kSuccess) {
          std::cerr << "CUTLASS error on line " << __LINE__ << std::endl;
          return result;
        }

      }
    }

    //
    // Stop profiling loop
    //

    // Record an event when the GEMM operations have been launched.
    result.error = cudaEventRecord(events[1]);
    if (result.error != cudaSuccess) {
      std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(result.error) << std::endl;
      return result;
    }
    
    //
    // Wait for work to be completed
    //

    result.error = cudaDeviceSynchronize();

    if (result.error != cudaSuccess)  {
      std::cerr << "Kernel execution error: " << cudaGetErrorString(result.error);
      return result;
    }

    // Wait for work on the device to complete.
    result.error = cudaEventSynchronize(events[1]);
    if (result.error != cudaSuccess) {
      std::cerr << "cudaEventSynchronize() failed: " << cudaGetErrorString(result.error) << std::endl;
      return result;
    }

    // Wait for work on the device to complete.
    result.error = cudaEventSynchronize(events[0]);
    if (result.error != cudaSuccess) {
      std::cerr << "cudaEventSynchronize() failed: " << cudaGetErrorString(result.error) << std::endl;
      return result;
    }

    // Measure elapsed runtime
    float runtime_ms = 0;
    result.error = cudaEventElapsedTime(&runtime_ms, events[0], events[1]);
    if (result.error != cudaSuccess) {
      std::cerr << "cudaEventElapsed() failed: " << cudaGetErrorString(result.error) << std::endl;
      return result;
    }

    // Compute average runtime and GFLOPs.
    result.runtime_ms = double(runtime_ms) / double(options.iterations);
    result.gflops = options.gflops(result.runtime_ms / 1000.0);

    //
    // Cleanup
    //

    for (auto event : events) {
      (void)cudaEventDestroy(event);
    }
    
    for (auto stream : cuda_streams) {
      if (stream) {
        (void)cudaStreamDestroy(stream);  
      }
    }

    std::cout << std::endl;
    std::cout << "Batched GEMM:\n"
      << "====================================================" << std::endl;

    std::cout << "    " << bin_problem_sizes.size() << " batched GEMMs launched" << std::endl;
    std::cout << std::endl;
    std::cout << "    " << "Batched Runtime: " << result.runtime_ms << " ms" << std::endl;
    std::cout << "    " << "Batched  GFLOPs: " << result.gflops << std::endl;

    if (options.output_file.good()) {
      options.output_file << options.output_tag << "," << provider << ",batched,"
        << problem_count() << "," << result.runtime_ms << "," << result.gflops << std::endl;
    }

    result.passed = true;
    return result;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char const **args) {

  //
  // This example uses mma.sync to directly access Tensor Cores to achieve peak performance.
  //

  cudaDeviceProp props;

  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (error != cudaSuccess) {
    std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
    return -1;
  }

  if (__CUDACC_VER_MAJOR__ < 11 || props.major < 8) {
  
    //
    // This example requires an NVIDIA Ampere-architecture GPU.
    //

    std::cout 
      << "CUTLASS's Grouped GEMM example requires a GPU of NVIDIA's Ampere Architecture or "
      << "later (compute capability 80 or greater).\n";

    return 0;
  }

  //
  // Parse options
  //

  Options options;
  
  options.parse(argc, args);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  if (options.error) {
    std::cerr << "Aborting execution." << std::endl;
    return -1;
  }

  //
  // Define the Grouped GEMM type
  //

  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = float;

  using LayoutA = cutlass::layout::ColumnMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::ColumnMajor;

  using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    cutlass::half_t, 
    LayoutA,
    cutlass::ComplexTransform::kNone,
    8,
    cutlass::half_t,
    LayoutB,
    cutlass::ComplexTransform::kNone,
    8,
    ElementOutput, LayoutC,
    ElementAccumulator, 
    cutlass::arch::OpClassTensorOp, 
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<
        ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
        ElementAccumulator, ElementAccumulator>,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle, 
    4>::GemmKernel;

  using GemmGrouped = cutlass::gemm::device::GemmGrouped<GemmKernel>;

  //
  // Define a conventional batched GEMM type
  //

  // Gemm operator cutlass_tensorop_f16_s16816gemm_f16_128x128_32x4_nt_align8
  using GemmBatched = cutlass::gemm::device::GemmUniversal<
    cutlass::half_t, LayoutA,
    cutlass::half_t, LayoutB,
    ElementOutput,   LayoutC,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<
      ElementOutput, 
      128 / cutlass::sizeof_bits<ElementOutput>::value,
      ElementAccumulator, 
      ElementAccumulator
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    4
  >;

  //
  // Profile it
  //

  TestbedGrouped<GemmGrouped, GemmBatched> testbed(options);

  if (!testbed.sufficient()) {
    std::cout << "The active CUDA device lacks sufficient hardware resources to execute this kernel.\n";
    return 0;
  }

  Result result = testbed.profile_grouped();
  if (!result.passed) {
    std::cout << "Profiling CUTLASS grouped GEMM has failed.\n";
    std::cout << "\nFailed\n";
    return -1;
  }

  result = testbed.profile_batched();
  if (!result.passed) {

    std::cout << "Profiling batched GEMM has failed.\n";
    std::cout << "\nFailed\n";
    return -1;
  }

  std::cout << "\nPassed\n";

  return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
