// Copyright (c) 2021 CINN Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/cinn/hlir/framework/instruction.h"

#include <fstream>
#include <sstream>

#include "paddle/cinn/common/test_helper.h"
#include "paddle/cinn/hlir/framework/accuracy_checker.h"
#include "paddle/cinn/runtime/flags.h"
#include "paddle/cinn/utils/profiler.h"

PD_DECLARE_bool(cinn_sync_run);
PD_DECLARE_string(cinn_self_check_accuracy);

namespace cinn {
namespace hlir {
namespace framework {

namespace details {
class ResultsPrint {
 public:
  static ResultsPrint* GetInstance() {
    static ResultsPrint print;
    return &print;
  }

  void write(const std::string& result_str) {
    if (of_.is_open()) {
      of_ << result_str << std::endl;
    } else if (!FLAGS_cinn_self_check_accuracy.empty()) {
      std::cerr << result_str << std::endl;
    } else {
      VLOG(2) << result_str << std::endl;
    }
  }

 private:
  ResultsPrint() {
    bool print_to_file =
        !FLAGS_cinn_self_check_accuracy.empty() &&
        !cinn::runtime::CheckStringFlagTrue(FLAGS_cinn_self_check_accuracy) &&
        !cinn::runtime::CheckStringFlagFalse(FLAGS_cinn_self_check_accuracy);

    if (print_to_file) {
      of_.open(FLAGS_cinn_self_check_accuracy, std::ios_base::out);
      if (of_.is_open()) {
        LOG(INFO) << "The CINN compute results will writing into file: \""
                  << FLAGS_cinn_self_check_accuracy << "\"";
      } else if (!FLAGS_cinn_self_check_accuracy.empty()) {
        LOG(WARNING) << "Failed to open file: \""
                     << FLAGS_cinn_self_check_accuracy
                     << "\", the CINN compute result will print.";
      }
    }
  }

  ~ResultsPrint() {
    if (of_.is_open()) {
      of_.close();
    }
  }

  std::ofstream of_;
};
}  // namespace details

void Instruction::UpdateArgsCache(
    const std::map<std::string, cinn_pod_value_t>* name2podargs) {
  int cache_size = size();
  args_cached_.resize(cache_size);

  for (int i = 0; i < cache_size; ++i) {
    common::ArgsBuilder builder;
    std::vector<std::string> all_args = in_args_[i];
    all_args.insert(
        std::end(all_args), out_args_[i].begin(), out_args_[i].end());

    if (name2podargs != nullptr) {
      for (const auto& arg : all_args) {
        CHECK_NE(name2podargs->count(arg), 0)
            << "Argument [" << arg << "] not found in the name2podargs";
        VLOG(5) << "Get a argument, name=" << arg
                << ",type_code=" << name2podargs->at(arg).type_code();
        builder.Add(name2podargs->at(arg));
      }
    } else {
      for (const auto& arg : all_args) {
        auto* var = scope_->FindVar(arg);
        CHECK(var) << "Argument [" << arg << "] not found in the scope";

        // TODO(Superjomn) Support other types.
        auto& tensor = absl::get<Tensor>(*var);
        VLOG(5) << "Get a argument, name=" << arg;
        builder.Add(tensor->buffer());
      }
    }

    args_cached_[i] = builder.Build();
  }
}

void Instruction::Finalize() {
  if (fn_ptrs_.size() > 1 && fn_ptrs_.size() != in_args_.size()) {
    out_args_.back()[0] = out_args_.front()[0];
    out_args_.erase(out_args_.begin());
    in_args_.erase(in_args_.begin());
  }

  finalized_flag_ = true;
}

void Instruction::Run(
    const std::map<std::string, cinn_pod_value_t>* name2podargs,
    bool dryrun,
    void* stream,
    bool use_cache) {
  utils::RecordEvent record_run(function_name_,
                                cinn::utils::EventType::kInstruction);
  CHECK(finalized_flag_) << "Instruction must be finalized before run";
  if (function_name_ == "no_run") {
    VLOG(2) << "skip instruction";
    return;
  }

  VLOG(2) << "Run function " << function_name_;

  {
    utils::RecordEvent record_args("UpdateArgsCache",
                                   cinn::utils::EventType::kInstruction);
    if (!use_cache || args_cached_.size() != size()) {
      UpdateArgsCache(name2podargs);
    }
  }

  utils::RecordEvent record_args("Instruction::Run",
                                 cinn::utils::EventType::kInstruction);
#if defined(CINN_WITH_CUDA) && !defined(CINN_WITH_CUDNN)
  if (function_name_ == "cublas_gemm" && target_.arch == Target::Arch::NVGPU) {
    auto& pod_args = args_cached_[0];
    VLOG(3) << "The pod_args size of cublas_gemm: " << pod_args.size();
    runtime::cuda::cinn_gpu_cublas_gemm(attrs,
                                        pod_args[0],
                                        pod_args[1],
                                        pod_args[2],
                                        pod_args[3],
                                        static_cast<cudaStream_t>(stream));
  } else if (function_name_ == "cublas_matmul" &&
             target_.arch == Target::Arch::NVGPU) {
    auto& pod_args = args_cached_[0];
    VLOG(3) << "The pod_args size of cublas_matmul: " << pod_args.size();
    runtime::cuda::cinn_gpu_cublas_gemm(attrs,
                                        pod_args[0],
                                        pod_args[1],
                                        nullptr,
                                        pod_args[2],
                                        static_cast<cudaStream_t>(stream));
  } else {
    VLOG(3) << "Runing extern function " << function_name_;
    for (int idx = 0; idx < fn_ptrs_.size(); ++idx) {
      VLOG(3) << "Runing func name: " << fn_names_[idx];
      auto& pod_args = args_cached_[idx];
      CHECK(fn_ptrs_[idx]) << "The LoweredFunc address should be set first by "
                              "calling SetLoweredFunc method";
      if (!dryrun) {
        if (target_ == common::DefaultNVGPUTarget()) {
          ((lower_func_ptr_g)fn_ptrs_[idx])(
              static_cast<void*>(pod_args.data()), pod_args.size(), stream);
        } else {
          ((lower_func_ptr_t)fn_ptrs_[idx])(static_cast<void*>(pod_args.data()),
                                            pod_args.size());
        }
      }
    }
    VLOG(3) << "Done Runing extern function " << function_name_;
  }
#elif defined(CINN_WITH_CUDNN)
  auto& pod_args = args_cached_[0];
  // Here conv2d and depthwise_conv2d are implemented by one cudnn api
  // cudnnConvolutionForward
  if ((function_name_ == "conv2d" || function_name_ == "depthwise_conv2d") &&
      target_.arch == Target::Arch::NVGPU) {
    if (str_attrs[0] == "forward") {
      if (str_attrs.size() > 1 && str_attrs[1] == "NHWC") {
        absl::flat_hash_map<std::string, int> attrs_map = {
            {"input_n", attrs[0]},     {"input_h", attrs[1]},
            {"input_w", attrs[2]},     {"input_c", attrs[3]},
            {"weights_n", attrs[4]},   {"weights_c", attrs[5]},
            {"weights_h", attrs[6]},   {"weights_w", attrs[7]},
            {"pad_h", attrs[8]},       {"pad_w", attrs[9]},
            {"stride_h", attrs[10]},   {"stride_w", attrs[11]},
            {"dilation_h", attrs[12]}, {"dilation_w", attrs[13]},
            {"groups", attrs[14]},     {"output_n", attrs[15]},
            {"output_h", attrs[16]},   {"output_w", attrs[17]},
            {"output_c", attrs[18]},
        };
        runtime::cuda::cinn_gpu_cudnn_conv2d(attrs_map,
                                             pod_args[0],
                                             pod_args[1],
                                             pod_args[2],
                                             static_cast<cudaStream_t>(stream),
                                             common::Layout::kNHWC);

      } else {
        absl::flat_hash_map<std::string, int> attrs_map = {
            {"input_n", attrs[0]},     {"input_c", attrs[1]},
            {"input_h", attrs[2]},     {"input_w", attrs[3]},
            {"weights_n", attrs[4]},   {"weights_c", attrs[5]},
            {"weights_h", attrs[6]},   {"weights_w", attrs[7]},
            {"pad_h", attrs[8]},       {"pad_w", attrs[9]},
            {"stride_h", attrs[10]},   {"stride_w", attrs[11]},
            {"dilation_h", attrs[12]}, {"dilation_w", attrs[13]},
            {"groups", attrs[14]},     {"output_n", attrs[15]},
            {"output_c", attrs[16]},   {"output_h", attrs[17]},
            {"output_w", attrs[18]},
        };
        runtime::cuda::cinn_gpu_cudnn_conv2d(attrs_map,
                                             pod_args[0],
                                             pod_args[1],
                                             pod_args[2],
                                             static_cast<cudaStream_t>(stream),
                                             common::Layout::kNCHW);
      }
    } else if (str_attrs[0] == "backward_data") {
      // w, dy, dx
      absl::flat_hash_map<std::string, int> attrs_map = {
          {"input_n", attrs[15]},    {"input_c", attrs[16]},
          {"input_h", attrs[17]},    {"input_w", attrs[18]},
          {"weights_n", attrs[0]},   {"weights_c", attrs[1]},
          {"weights_h", attrs[2]},   {"weights_w", attrs[3]},
          {"pad_h", attrs[8]},       {"pad_w", attrs[9]},
          {"stride_h", attrs[10]},   {"stride_w", attrs[11]},
          {"dilation_h", attrs[12]}, {"dilation_w", attrs[13]},
          {"groups", attrs[14]},     {"output_n", attrs[4]},
          {"output_c", attrs[5]},    {"output_h", attrs[6]},
          {"output_w", attrs[7]},
      };
      // w, dy, dx
      runtime::cuda::cinn_gpu_cudnn_conv2d_backward_data(
          attrs_map,
          pod_args[0],
          pod_args[1],
          pod_args[2],
          static_cast<cudaStream_t>(stream));
    } else {
      // x, dy, w
      absl::flat_hash_map<std::string, int> attrs_map = {
          {"input_n", attrs[0]},     {"input_c", attrs[1]},
          {"input_h", attrs[2]},     {"input_w", attrs[3]},
          {"weights_n", attrs[15]},  {"weights_c", attrs[16]},
          {"weights_h", attrs[17]},  {"weights_w", attrs[18]},
          {"pad_h", attrs[8]},       {"pad_w", attrs[9]},
          {"stride_h", attrs[10]},   {"stride_w", attrs[11]},
          {"dilation_h", attrs[12]}, {"dilation_w", attrs[13]},
          {"groups", attrs[14]},     {"output_n", attrs[4]},
          {"output_c", attrs[5]},    {"output_h", attrs[6]},
          {"output_w", attrs[7]},
      };
      // x, dy, w
      runtime::cuda::cinn_gpu_cudnn_conv2d_backward_filter(
          attrs_map,
          pod_args[0],
          pod_args[1],
          pod_args[2],
          static_cast<cudaStream_t>(stream));
    }
  } else if (function_name_ == "pool2d" &&
             target_.arch == Target::Arch::NVGPU) {
    runtime::cuda::cinn_gpu_cudnn_pool2d(attrs,
                                         str_attrs,
                                         pod_args[0],
                                         pod_args[1],
                                         static_cast<cudaStream_t>(stream));
  } else if (function_name_ == "softmax" &&
             target_.arch == Target::Arch::NVGPU) {
    CHECK_EQ(pod_args.size(), 3);
    runtime::cuda::cinn_gpu_cudnn_softmax(
        attrs, pod_args[0], pod_args[1], static_cast<cudaStream_t>(stream));
  } else if (function_name_ == "mul" && target_.arch == Target::Arch::NVGPU) {
    CHECK_EQ(pod_args.size(), 4);
    runtime::cuda::cinn_gpu_cublas_mul(attrs,
                                       pod_args[0],
                                       pod_args[1],
                                       pod_args[2],
                                       static_cast<cudaStream_t>(stream));
  } else if (function_name_ == "cublas_gemm" &&
             target_.arch == Target::Arch::NVGPU) {
    VLOG(3) << "The pod_args size of cublas_gemm: " << pod_args.size();
    runtime::cuda::cinn_gpu_cublas_gemm(attrs,
                                        pod_args[0],
                                        pod_args[1],
                                        pod_args[2],
                                        pod_args[3],
                                        static_cast<cudaStream_t>(stream));
  } else if (function_name_ == "cublas_matmul" &&
             target_.arch == Target::Arch::NVGPU) {
    auto& pod_args = args_cached_[0];
    VLOG(3) << "The pod_args size of cublas_matmul: " << pod_args.size();
    runtime::cuda::cinn_gpu_cublas_gemm(attrs,
                                        pod_args[0],
                                        pod_args[1],
                                        nullptr,
                                        pod_args[2],
                                        static_cast<cudaStream_t>(stream));
  } else {
    VLOG(3) << "Runing extern function " << function_name_;
    for (int idx = 0; idx < fn_ptrs_.size(); ++idx) {
      VLOG(3) << "Runing func name: " << fn_names_[idx];
      auto& pod_args = args_cached_[idx];
      CHECK(fn_ptrs_[idx]) << "The LoweredFunc address should be set first by "
                              "calling SetLoweredFunc method";
      if (!dryrun) {
        if (target_ == common::DefaultNVGPUTarget()) {
          ((lower_func_ptr_g)fn_ptrs_[idx])(
              static_cast<void*>(pod_args.data()), pod_args.size(), stream);
        } else {
          ((lower_func_ptr_t)fn_ptrs_[idx])(static_cast<void*>(pod_args.data()),
                                            pod_args.size());
        }
      }
    }
    VLOG(3) << "Done Runing extern function " << function_name_;
  }
#else
  VLOG(3) << "Runing extern function " << function_name_;
  for (int idx = 0; idx < fn_ptrs_.size(); ++idx) {
    VLOG(3) << "Runing func name: " << fn_names_[idx];
    auto& pod_args = args_cached_[idx];
    CHECK(fn_ptrs_[idx]) << "The LoweredFunc address should be set first by "
                            "calling SetLoweredFunc method";
    if (!dryrun) {
      if (target_ == common::DefaultNVGPUTarget()) {
        ((lower_func_ptr_g)fn_ptrs_[idx])(
            static_cast<void*>(pod_args.data()), pod_args.size(), stream);
      } else {
        ((lower_func_ptr_t)fn_ptrs_[idx])(static_cast<void*>(pod_args.data()),
                                          pod_args.size());
      }
    }
  }
  VLOG(3) << "Done Runing extern function " << function_name_;
#endif

  if (!cinn::runtime::CheckStringFlagFalse(FLAGS_cinn_self_check_accuracy)) {
    CheckResults(name2podargs, stream);
  }
  // TODO(thisjiang): revert while flags correct
  //   if (FLAGS_cinn_sync_run) {
  // #ifdef CINN_WITH_CUDA
  //     utils::RecordEvent record_sync("FLAGS_cinn_sync_run");
  //     CUDA_CALL(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)));
  // #endif
  //   }
}

std::string Instruction::DumpInstruction() const {
  std::stringstream ss;
  ss << "Instruction {" << std::endl;
  for (size_t i = 0; i < fn_names_.size(); ++i) {
    ss << "  Function " << fn_names_[i] << ":" << std::endl;
    ss << "    function ptr: " << fn_ptrs_[i] << std::endl;

    auto in_arg = in_args_[i];
    std::sort(in_arg.begin(), in_arg.end());
    for (auto& in_name : in_arg) {
      ss << "    input: " << in_name << std::endl;
    }

    auto out_arg = out_args_[i];
    std::sort(out_arg.begin(), out_arg.end());
    for (auto& out_name : out_arg) {
      ss << "    output: " << out_name << std::endl;
    }
  }
  ss << "}" << std::endl;
  return ss.str();
}

void Instruction::CheckResults(
    const std::map<std::string, cinn_pod_value_t>* name2podargs, void* stream) {
#ifdef CINN_WITH_CUDA
  cudaStreamSynchronize(static_cast<cudaStream_t>(stream));
#endif

  if (fn_names_.size() == 1) {
    std::unordered_set<std::string> skipped_instr_set = {
        "malloc_buffer_instruction", "free_buffer_instruction"};
    for (auto& name : skipped_instr_set) {
      if (fn_names_[0].find(name) != std::string::npos) {
        // Skip the malloc & free buffer instructions.
        return;
      }
    }
  }

  AccuracyChecker checker(target_, scope_);

  std::stringstream ss;
  ss << "Instruction {" << std::endl;
  for (size_t i = 0; i < fn_names_.size(); ++i) {
    ss << "  Function " << fn_names_[i] << ":" << std::endl;

    auto in_arg = in_args_[i];
    std::sort(in_arg.begin(), in_arg.end());
    for (auto& in_name : in_arg) {
      std::string result_str;
      if (name2podargs) {
        result_str = checker(name2podargs, in_name);
      } else {
        result_str = checker(in_name);
      }
      ss << "    input: " << result_str << std::endl;
    }

    auto out_arg = out_args_[i];
    std::sort(out_arg.begin(), out_arg.end());
    for (auto& out_name : out_arg) {
      std::string result_str;
      if (name2podargs) {
        result_str = checker(name2podargs, out_name);
      } else {
        result_str = checker(out_name);
      }
      ss << "    output: " << result_str << std::endl;
    }
  }
  ss << "}" << std::endl;

  details::ResultsPrint::GetInstance()->write(ss.str());
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
