// Copyright (c) 2022 CINN Authors. All Rights Reserved.
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

#include <string.h>

#ifdef CINN_WITH_CUDA
#include <cuda_runtime.h>
#endif

#include "paddle/cinn/runtime/custom_function.h"
#include "paddle/cinn/runtime/flags.h"
#include "paddle/cinn/utils/string.h"

PD_DECLARE_string(cinn_check_fusion_accuracy_pass);

namespace cinn {
namespace runtime {

using common::Target;
using hlir::framework::Shape;
using hlir::framework::Tensor;

namespace utils {
void AssertTrueMsgTool::SetMsg(int key, const std::string& msg) {
  global_msg_[key] = msg;
}

const std::string& AssertTrueMsgTool::GetMsg(int key) {
  CHECK(global_msg_.find(key) != global_msg_.end())
      << "Cannot find assert_true message key " << key;
  return global_msg_[key];
}

void AssertTrueMsgTool::InitFlagInfo() {
  // only need parse flag once
  if (!flag_values_.empty()) {
    return;
  }
  // default value
  flag_values_ = {{"only_warning", false},
                  {"rtol", 1e-5f},
                  {"atol", 1e-8f},
                  {"equal_nan", false}};
  if (CheckStringFlagFalse(FLAGS_cinn_check_fusion_accuracy_pass) ||
      CheckStringFlagTrue(FLAGS_cinn_check_fusion_accuracy_pass)) {
    // using default value
    LOG(INFO) << "The FLAGS_cinn_check_fusion_accuracy_pass will check fusion "
                 "group accuracy with: "
                 "\"only_warning=false;rtol=1e-5;atol=1e-8;equal_nan=false\"";
    return;
  }

  // parse flags
  const auto& args =
      cinn::utils::Split(FLAGS_cinn_check_fusion_accuracy_pass, ";");
  for (const auto& str : args) {
    if (str.empty()) {
      continue;
    }
    const auto& flag_arg = cinn::utils::Split(str, "=");
    CHECK_EQ(flag_arg.size(), 2UL)
        << "The FLAGS_cinn_check_fusion_accuracy_pass must be the format of "
           "\"only_warning=false;rtol=1e-5;atol=1e-8;equal_nan=false\"";

    if (flag_arg[0] == "only_warning" || flag_arg[0] == "equal_nan") {
      // bool type parameter
      flag_values_[flag_arg[0]] = CheckStringFlagTrue(flag_arg[1]);
    } else if (flag_arg[0] == "rtol" || flag_arg[0] == "atol") {
      // string type parameter
      flag_values_[flag_arg[0]] = std::stof(flag_arg[1]);
    } else {
      LOG(FATAL)
          << "The FLAGS_cinn_check_fusion_accuracy_pass only support parameter "
             "\"only_warning/rtol/atol/equal_nan\" now";
    }
  }

  LOG(INFO) << "The FLAGS_cinn_check_fusion_accuracy_pass will check fusion "
               "group accuracy with: \""
            << "only_warning="
            << cinn::utils::Attribute2String(flag_values_.at("only_warning"))
            << ";rtol="
            << cinn::utils::Attribute2String(flag_values_.at("rtol"))
            << ";atol="
            << cinn::utils::Attribute2String(flag_values_.at("atol"))
            << ";equal_nan="
            << cinn::utils::Attribute2String(flag_values_.at("equal_nan"))
            << "\"";
}

bool MemcpyToHost(void* dst,
                  const void* src,
                  size_t bytes,
                  const Target& input_target,
                  void* stream = nullptr) {
  if (input_target == common::DefaultNVGPUTarget()) {
#ifdef CINN_WITH_CUDA
    const auto& cuda_stream = static_cast<cudaStream_t>(stream);
    cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, cuda_stream);
    cudaStreamSynchronize(cuda_stream);
    return true;
#else
    LOG(FATAL)
        << "NVGPU Target only support on flag CINN_WITH_CUDA ON! Please check.";
    return false;
#endif
  }
  if (input_target == common::DefaultHostTarget()) {
    memcpy(dst, src, bytes);
    return true;
  }
  LOG(FATAL) << "MemcpyToHost Only support cpu or nvgpu -> cpu, but here the "
                "input target is "
             << input_target << "! Please check.";
  return false;
}

bool MemcpyToDevice(void* dst,
                    const void* src,
                    size_t bytes,
                    const Target& input_target,
                    void* stream = nullptr) {
#ifdef CINN_WITH_CUDA
  if (input_target == common::DefaultNVGPUTarget()) {
    cudaMemcpyAsync(dst,
                    src,
                    bytes,
                    cudaMemcpyDeviceToDevice,
                    static_cast<cudaStream_t>(stream));
    return true;
  } else if (input_target == common::DefaultHostTarget()) {
    cudaMemcpyAsync(dst,
                    src,
                    bytes,
                    cudaMemcpyHostToDevice,
                    static_cast<cudaStream_t>(stream));
    return true;
  } else {
    LOG(FATAL) << "MemcpyToDevice only support cpu or nvgpu -> nvgpu, but here "
                  "the input target is "
               << input_target << "! Please check.";
    return false;
  }
#else
  LOG(FATAL) << "MemcpyToDevice only support nvgpu, and NVGPU Target only "
                "support when flag CINN_WITH_CUDA ON! Please check.";
  return false;
#endif
}
}  // namespace utils

void CheckAssertTrue(const bool* x,
                     const size_t numel,
                     bool only_warning,
                     const std::string& msg,
                     const Target& target) {
  // check false number and first false offset
  int error_num = 0, first_diff = -1;
  for (int i = 0; i < numel; ++i) {
    if (!x[i]) {
      ++error_num;
      if (first_diff == -1) {
        first_diff = i;
      }
    }
  }

  // raise error information
  if (error_num > 0) {
    std::string error_info = "[AssertTrue] Check failed!\n";
    error_info += "- target: " + target.arch_str() + "\n";
    error_info += "- assert false number: " + std::to_string(error_num) + "\n";
    error_info += "- first false offset: " + std::to_string(first_diff) + "\n";
    error_info += "- group message:\n" + msg;

    if (only_warning) {
      LOG(WARNING) << error_info;
    } else {
      LOG(FATAL) << error_info;
    }
  } else {
    VLOG(1) << "[AssertTrue] Check succeed!\n"
            << "- group message:\n" + msg;
  }
}

void cinn_assert_true(void* v_args,
                      int num_args,
                      int msg,
                      bool only_warning,
                      void* stream,
                      const Target& target) {
  // why x->type and output->type are empty?
  // CHECK(x->type == cinn_bool_t()) << "The input type of AssertTrue should be
  // bool, but here " << x->type.bits
  //                                 << "! Please check.";
  // CHECK(output->type == cinn_bool_t()) << "The output type of AssertTrue
  // should be bool, but here " << output->type.bits
  //                                      << "! Please check.";

  cinn_pod_value_t* args = static_cast<cinn_pod_value_t*>(v_args);

  cinn_buffer_t* x = args[0].operator cinn_buffer_t*();
  cinn_buffer_t* output = args[1].operator cinn_buffer_t*();

  // create cpu tensor
  std::vector<int> shape;
  shape.resize(x->dimensions);
  for (int i = 0; i < shape.size(); ++i) {
    shape[i] = x->dims[i];
  }

  Tensor cpu_tensor;
  cpu_tensor->Resize(Shape(shape));
  bool* dst = cpu_tensor->mutable_data<bool>(common::DefaultHostTarget());

  // copy data from gpu to cpu
  const bool* src = reinterpret_cast<const bool*>(x->memory);
  size_t numel = cpu_tensor->shape().numel();
  utils::MemcpyToHost(dst, src, numel * sizeof(bool), target, stream);

  CheckAssertTrue(dst,
                  numel,
                  only_warning,
                  utils::AssertTrueMsgTool::GetInstance()->GetMsg(msg),
                  target);

  if (target == common::DefaultNVGPUTarget()) {
    utils::MemcpyToDevice(
        output->memory, x->memory, numel * sizeof(bool), target, stream);
  } else {
    utils::MemcpyToHost(
        output->memory, x->memory, numel * sizeof(bool), target, stream);
  }
}

}  // namespace runtime
}  // namespace cinn
