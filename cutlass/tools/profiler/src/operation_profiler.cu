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
/* \file
   \brief Defines a math function
*/

#include <algorithm>
#include <stdexcept>
#include <iomanip>
#include <cstring>
#include <fstream>
#include <sstream>

#ifdef __unix__
#include <unistd.h>
#elif defined(_WIN32) || defined(WIN32)
#include <windows.h>
#else
// sleep not supported
#endif

#include "options.h"
#include "operation_profiler.h"
#include "gpu_timer.h"

///////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace profiler {

///////////////////////////////////////////////////////////////////////////////////////////////////

OperationProfiler::OperationProfiler(): kind_(library::OperationKind::kInvalid) { }

/// Ctor
OperationProfiler::OperationProfiler(
  Options const &options,
  library::OperationKind kind,
  ArgumentDescriptionVector const &arguments,
  ProviderVector const & verification_providers
): 
  kind_(kind), arguments_(arguments) {

  ArgumentDescriptionVector tile_description_arguments{
    {ArgumentTypeID::kEnumerated, {"op_class", "opcode-class"}, "Class of math instruction (simt, tensorop, wmmatensorop, wmma)"},
    {ArgumentTypeID::kEnumerated, {"accum", "accumulator-type"}, "Math instruction accumulator data type"},
    {ArgumentTypeID::kInteger, {"cta_m", "threadblock-shape::m"}, "Threadblock shape in the M dimension"},
    {ArgumentTypeID::kInteger, {"cta_n", "threadblock-shape::n"}, "Threadblock shape in the N dimension"},
    {ArgumentTypeID::kInteger, {"cta_k", "threadblock-shape::k"}, "Threadblock shape in the K dimension"},
    {ArgumentTypeID::kInteger, {"stages", "threadblock-stages"}, "Number of stages of threadblock-scoped matrix multiply"},
    {ArgumentTypeID::kInteger, {"warps_m", "warp-count::m"}, "Number of warps within threadblock along the M dimension"},
    {ArgumentTypeID::kInteger, {"warps_n", "warp-count::n"}, "Number of warps within threadblock along the N dimension"},
    {ArgumentTypeID::kInteger, {"warps_k", "warp-count::k"}, "Number of warps within threadblock along the K dimension"},
    {ArgumentTypeID::kInteger, {"inst_m", "instruction-shape::m"}, "Math instruction shape in the M dimension"},
    {ArgumentTypeID::kInteger, {"inst_n", "instruction-shape::n"}, "Math instruction shape in the N dimension"},
    {ArgumentTypeID::kInteger, {"inst_k", "instruction-shape::k"}, "Math instruction shape in the K dimension"},
    {ArgumentTypeID::kInteger, {"min_cc", "minimum-compute-capability"}, "Minimum device compute capability"},
    {ArgumentTypeID::kInteger, {"max_cc", "maximum-compute-capability"}, "Maximum device compute capability"}
  };

  arguments_.insert(arguments_.end(), tile_description_arguments.begin(), tile_description_arguments.end());

  for (auto provider : verification_providers) {
    if (std::find(
      options.verification.providers.begin(), 
      options.verification.providers.end(), 
      provider) != options.verification.providers.end()) {

      verification_providers_.push_back(provider);
    }
  }
}

/// Destructor
OperationProfiler::~OperationProfiler() {

}

/// Gets the schema description
std::string const & OperationProfiler::description() const {
  return description_;
}

/// Prints usage statement for the math function
void OperationProfiler::print_usage(std::ostream &out) const {
  for (auto const & desc : arguments_) {

    size_t const kAliasStart = 10;

    size_t columns = 0;
    
    std::string type_str = to_string(desc.type);
    columns += type_str.size();

    out << "  [" << type_str << "]";

    if (columns < kAliasStart) {
      out << std::string(kAliasStart - columns, ' ');  
    }

    columns = 0;

    int j = 0;
    for (auto const & alias : desc.aliases) {
      columns += alias.size() + (j ? 1 : 0) + 2;

      out << (j++ ? "," : "") << "--" << alias;
    }

    size_t const kTotalColumns = 50;

    if (columns < kTotalColumns) {
      out << std::string(kTotalColumns - columns, ' ');
    }

    out << desc.description << "\n";
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Returns true if the current operation description satisfies the problem space
bool OperationProfiler::satisfies(
  library::OperationDescription const &op_desc,
  ProblemSpace const &problem_space,
  ProblemSpace::Problem const &problem) {

  library::OpcodeClassID opcode_class;
  if (arg_as_OpcodeClassID(opcode_class, "op_class", problem_space, problem)) {
    if (opcode_class != op_desc.tile_description.math_instruction.opcode_class) {
      return false;
    }
  }
  
  int64_t int_value;

  if (arg_as_int(int_value, "inst_m", problem_space, problem)) {
    if (int64_t(op_desc.tile_description.math_instruction.instruction_shape.m()) != int_value) {
      return false;
    }
  }

  if (arg_as_int(int_value, "inst_n", problem_space, problem)) {
    if (int64_t(op_desc.tile_description.math_instruction.instruction_shape.n()) != int_value) {
      return false;
    }
  }

  if (arg_as_int(int_value, "inst_k", problem_space, problem)) {
    if (int64_t(op_desc.tile_description.math_instruction.instruction_shape.k()) != int_value) {
      return false;
    }
  }

  if (arg_as_int(int_value, "cta_m", problem_space, problem)) {
    if (int64_t(op_desc.tile_description.threadblock_shape.m()) != int_value) {
      return false;
    }
  }

  if (arg_as_int(int_value, "cta_n", problem_space, problem)) {
    if (int64_t(op_desc.tile_description.threadblock_shape.n()) != int_value) {
      return false;
    }
  }

  if (arg_as_int(int_value, "cta_k", problem_space, problem)) {
    if (int64_t(op_desc.tile_description.threadblock_shape.k()) != int_value) {
      return false;
    }
  }

  if (arg_as_int(int_value, "stages", problem_space, problem)) {
    if (int64_t(op_desc.tile_description.threadblock_stages) != int_value) {
      return false;
    }
  }

  if (arg_as_int(int_value, "warps_m", problem_space, problem)) {
    if (int64_t(op_desc.tile_description.warp_count.m()) != int_value) {
      return false;
    }
  }

  if (arg_as_int(int_value, "warps_n", problem_space, problem)) {
    if (int64_t(op_desc.tile_description.warp_count.n()) != int_value) {
      return false;
    }
  }

  if (arg_as_int(int_value, "warps_k", problem_space, problem)) {
    if (int64_t(op_desc.tile_description.warp_count.k()) != int_value) {
      return false;
    }
  }

  library::NumericTypeID numeric_type;
  if (arg_as_NumericTypeID(numeric_type, "accum", problem_space, problem)) {
    if (numeric_type != op_desc.tile_description.math_instruction.element_accumulator) {
      return false;
    }
  }

  return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Entry point to profile all operations in the manifest
int OperationProfiler::profile_all(
  Options const &options, 
  library::Manifest const &manifest, 
  DeviceContext &device_context) {
  
  ProblemSpace problem_space(arguments_, options.cmdline);

  // 1. Construct performance report
  PerformanceReport report(options, problem_space.argument_names(), kind_);

  // 2. For each problem in problem space
  ProblemSpace::Iterator problem_it = problem_space.begin();
  ProblemSpace::Iterator problem_end = problem_space.end();

  bool continue_profiling = true, internal_error = false;

  // For each problem in problem space
  for (; continue_profiling && problem_it != problem_end; ++problem_it) {

    ProblemSpace::Problem problem = problem_it.at();

    report.next_problem();

    // For each operation in manifest
    for (auto const & operation_ptr : manifest) {

      library::Operation const *operation = operation_ptr.get();

      auto min_cc = operation->description().tile_description.minimum_compute_capability;
      auto max_cc = operation->description().tile_description.maximum_compute_capability;

      // Clear named allocations
      device_context.free();

      // Execute compatible cutlass operations if they satisfy the current device's compute capability
      if (operation->description().kind == kind_ &&
        operation->description().provider == library::Provider::kCUTLASS &&
        options.device.compute_capability() >= min_cc &&
          options.device.compute_capability() <= max_cc) {

        std::string operation_name(operation->description().name);

        // Filter kernels by name
        bool filtered_by_name = options.operation_names.empty();
        if (!filtered_by_name) {
          
          for (auto const & op_name : options.operation_names) {
            if (find_string_matches_(op_name, operation_name)) {
              filtered_by_name = true;
              break;
            }
          } 
        }

        for (auto const & op_name : options.excluded_operation_names) {
          if (find_string_matches_(op_name, operation_name)) {
            filtered_by_name = false;
            break;
          }
        }

        if (!filtered_by_name || !satisfies(operation->description(), problem_space, problem)) {
          continue;
        }
      
        // A. Initialize configuration
        Status status = this->initialize_configuration(
          options,
          report,
          device_context,
          operation,
          problem_space,
          problem);

        if (status == Status::kErrorInternal) {
          
          // If there was an internal error, consume the CUDA error and move to the next operation.
          (void)cudaGetLastError();
          
          report.append_results(results_);
          continue;
        }
        else if (status != Status::kSuccess) {
          // If the workspace could not be initialized for any other reason, continue to
          // the next operation.
          continue;
        }

        if (continue_profiling) {

          status = this->initialize_workspace(
            options,
            report,
            device_context,
            operation,
            problem_space,
            problem);

          if (status == Status::kErrorInternal) {

            // If there was an internal error, consume the CUDA error and move to the next operation.
            (void)cudaGetLastError();

            report.append_results(results_);
            continue;
          }
          else if (status != Status::kSuccess) {
            // If the workspace could not be initialized for any other reason, continue to
            // the next operation.
            continue;
          }
        }

        //
        // Profile CUTLASS if it is enabled
        //

        // B. Verify CUTLASS
         
        if (continue_profiling && options.profiling.provider_enabled(library::Provider::kCUTLASS)) {

          continue_profiling = this->verify_cutlass(
            options,
            report, 
            device_context, 
            operation, 
            problem_space,
            problem);
        }

        if (options.execution_mode == ExecutionMode::kDryRun) {
          report.append_results(results_);
          results_.clear();
          continue;
        }

        //
        // C. Optionally save workspace
        //

        if (options.verification.save_workspace == SaveWorkspace::kAlways) {
          save_workspace(
            device_context,
            options,
            operation->description(),
            library::Provider::kCUTLASS);
        }

        //
        // D. Profile
        //

        if (continue_profiling && options.profiling.enabled) {

          continue_profiling = this->profile(
            options, 
            report, 
            device_context, 
            operation, 
            problem_space,
            problem);
        }

        report.append_results(results_);
        results_.clear();
      }

      if (!continue_profiling) {
        break;
      }
    } 
  }

  return internal_error ? 1 : 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Sleep for a given duration in ms
void OperationProfiler::sleep(int sleep_duration) {
  if (sleep_duration) {
    #ifdef __unix__
    usleep(sleep_duration * 1000);
    #elif defined(_WIN32) || defined(WIN32)
    SleepEx(sleep_duration, false);
    #else
    // sleep not supported
    #endif 
  }
}


/// Compares tensors for equality
Disposition OperationProfiler::compare_tensors(
  Options const &options,
  DeviceAllocation &experimental,
  DeviceAllocation &reference,
  int64_t count) {

  if (experimental.type() != reference.type()) {
    return Disposition::kIncorrect;
  }

  bool passed = false;

  if (count == 0) {
    count = reference.capacity();
  }

  if (options.verification.epsilon == 0) {

    // bit-level equality
    passed = DeviceAllocation::block_compare_equal(
      experimental.type(), 
      experimental.data(),
      reference.data(),
      count);
  }
  else {

    // relative error function
    passed = DeviceAllocation::block_compare_relatively_equal(
      experimental.type(), 
      experimental.data(),
      reference.data(),
      count,
      options.verification.epsilon,
      options.verification.nonzero_floor);
  }

  return passed ? Disposition::kPassed : Disposition::kIncorrect;
}

/// Saves the workspace
void OperationProfiler::save_workspace(
  DeviceContext &device_context,
  Options const &options,
  library::OperationDescription const &desc,
  library::Provider provider,
  library::Provider verification_provider) {

  for (auto const & named_allocation : device_context) {

    DeviceAllocation *allocation = named_allocation.second;
    
    std::stringstream filename;

    filename << desc.name << "_" << library::to_string(provider) << "_";

    if (verification_provider != library::Provider::kInvalid) {
      filename << "verified_by_" << library::to_string(verification_provider) << "_";
    }

    filename << named_allocation.first + ".mat";

    std::ofstream out(filename.str());

    allocation->write_tensor_csv(out);
    out << "\n";

    if (options.report.verbose) {
      std::cout << "wrote '" << filename.str() << "'" << std::endl;
    }
  } 
}


///////////////////////////////////////////////////////////////////////////////////////////////////

/// Method to profile a CUTLASS Operation
Status OperationProfiler::profile_cutlass_(
  double &runtime,
  Options const &options,
  library::Operation const *operation,
  void *arguments,
  void *host_workspace,
  void *device_workspace) {

  GpuTimer timer;

  //
  // Optional sleep to limit power consumption and thermals
  //

  sleep(options.profiling.sleep_duration);

  //
  // Warmup loop
  //

  Status status;

  for (int iteration = 0; iteration < options.profiling.warmup_iterations; ++iteration) {

    status = operation->run(
      arguments,
      host_workspace,
      device_workspace);

    if (status != Status::kSuccess) {
      return status;
    }
  }
  
  //
  // Initialize GPU timer
  //

  timer.start();

  //
  // Profiling loop
  //

  int Iterations = options.profiling.iterations;

  int iteration = 0;
  for (; iteration < Iterations; ++iteration) {
    
    status = operation->run(
      arguments,
      host_workspace,
      device_workspace);

    if (status != Status::kSuccess) {
      return status;
    }
  }

  //
  // Wait for completion
  //

  timer.stop_and_wait();

  //
  // Update performance result
  //
  
  runtime = timer.duration(iteration);

  return status;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Sets operation description 
void OperationProfiler::initialize_result_(
  PerformanceResult &result,
  library::OperationDescription const &operation_desc,
  ProblemSpace const &problem_space) {

  set_argument(result, "op_class", problem_space,
    library::to_string(operation_desc.tile_description.math_instruction.opcode_class));

  set_argument(result, "accum", problem_space,
    library::to_string(operation_desc.tile_description.math_instruction.element_accumulator));

  set_argument(result, "cta_m", problem_space, operation_desc.tile_description.threadblock_shape.m());
  set_argument(result, "cta_n", problem_space, operation_desc.tile_description.threadblock_shape.n());
  set_argument(result, "cta_k", problem_space, operation_desc.tile_description.threadblock_shape.k());
  set_argument(result, "stages", problem_space, operation_desc.tile_description.threadblock_stages);
  set_argument(result, "warps_m", problem_space, operation_desc.tile_description.warp_count.m());
  set_argument(result, "warps_n", problem_space, operation_desc.tile_description.warp_count.n());
  set_argument(result, "warps_k", problem_space, operation_desc.tile_description.warp_count.k());
  set_argument(result, "inst_m", problem_space, operation_desc.tile_description.math_instruction.instruction_shape.m());
  set_argument(result, "inst_n", problem_space, operation_desc.tile_description.math_instruction.instruction_shape.n());
  set_argument(result, "inst_k", problem_space, operation_desc.tile_description.math_instruction.instruction_shape.k());
  set_argument(result, "min_cc", problem_space, operation_desc.tile_description.minimum_compute_capability);
  set_argument(result, "max_cc", problem_space, operation_desc.tile_description.maximum_compute_capability);
}

/// Helper
void OperationProfiler::set_argument(
  PerformanceResult &result,
  char const *name,
  ProblemSpace const &problem_space,
  std::string const &value) {

  result.arguments.at(problem_space.argument_index(name)) = make_pair(std::string(name), value);
}

void OperationProfiler::set_argument(  
  PerformanceResult &result,
  char const *name,
  ProblemSpace const &problem_space,
  int64_t value) {

  result.arguments.at(problem_space.argument_index(name)) = make_pair(std::string(name), library::lexical_cast(value));
}


/// finds string matches filter_string in operation_name
bool OperationProfiler::find_string_matches_(
  std::string const &filter_string, 
  std::string const &operation_name) {
  // Returns true if all substrings appear in the operation_name in order
  
  // Split filter_string of the format "gemm*f32*nt" to tokens ["gemm", "f32", "nt"]
  std::string item;  
  std::istringstream iss(filter_string);
  std::vector<std::string> filter_tokens;
  while (std::getline(iss, item, '*')) {
    filter_tokens.push_back(item);
  }

  // Search filter_tokens in operation_name in order
  size_t start = 0, idx = 0;
  for(auto & token : filter_tokens) {
    // Check if characters left to be parsed in operation_name
    if (start < operation_name.length()) {
      // Find token in operation_name[start:]
      idx = operation_name.substr(start).find(token);
      if (idx == std::string::npos) {
        return false;
      }
    }
    start += (idx + token.length()); 
  }

  // All tokens in filter_string found in operation_name
  return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace profiler
} // namespace cutlass

///////////////////////////////////////////////////////////////////////////////////////////////////
