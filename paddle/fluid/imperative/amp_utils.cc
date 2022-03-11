// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/imperative/amp_utils.h"

namespace paddle {
namespace imperative {

// According to the input `place` and `dtype`, this function returns a tuple
// consists of three sets:
// 1) All operators registered in the Paddle framework.
// 2) All operators supported for `place` and `dtype`.
// 3) All operators unsupported for `place` and `dtype`.
// The input `place` is a type of string, which can only be `GPU` or `CPU`.
// The input `dtype` is a type of paddle::framework::proto::VarType::Type,
// which can be paddle::framework::proto::VarType::FP16,
// paddle::framework::proto::VarType::FP32 and so on.
std::tuple<std::unordered_set<std::string>, std::unordered_set<std::string>,
           std::unordered_set<std::string>>
OpSupportedInfos(const std::string& place,
                 framework::proto::VarType::Type dtype) {
  std::string query_place;
  std::transform(place.begin(), place.end(), std::back_inserter(query_place),
                 [](unsigned char c) { return std::toupper(c); });
  using fn_type = std::add_pointer<bool(const platform::Place&)>::type;
  std::unordered_map<std::string, fn_type> is_target_place{
      {"GPU", &platform::is_gpu_place}, {"CPU", &platform::is_cpu_place},
      {"XPU", &platform::is_xpu_place}, {"NPU", &platform::is_npu_place},
      {"MLU", &platform::is_mlu_place},
  };
  PADDLE_ENFORCE_NE(is_target_place.count(query_place), 0,
                    platform::errors::InvalidArgument(
                        "The argument `place` should be 'GPU', 'CPU', 'XPU', "
                        "'NPU', 'MLU', but got '%s'.",
                        place));

  std::unordered_set<std::string> all_ops;
  const auto& op_info = framework::OpInfoMap::Instance().map();
  for (auto it = op_info.begin(); it != op_info.end(); it++) {
    all_ops.emplace(it->first);
  }

  std::unordered_set<std::string> supported_ops;
  auto& all_kernels = framework::OperatorWithKernel::AllOpKernels();
  for (auto it = all_kernels.begin(); it != all_kernels.end(); it++) {
    for (auto& kernel_type : it->second) {
      if (is_target_place[query_place](kernel_type.first.place_) &&
          kernel_type.first.data_type_ == dtype) {
        supported_ops.emplace(it->first);
      }
    }
  }

  auto phi_kernels = phi::KernelFactory::Instance().kernels();
  for (auto& kernel_pair : phi_kernels) {
    auto op_type = phi::TransToFluidOpName(kernel_pair.first);
    for (auto& info_pair : kernel_pair.second) {
      framework::OpKernelType kernel_type =
          framework::TransPhiKernelKeyToOpKernelType(info_pair.first);
      if (is_target_place[query_place](kernel_type.place_) &&
          kernel_type.data_type_ == dtype && all_ops.count(op_type)) {
        VLOG(4) << op_type << " " << supported_ops.size();
        supported_ops.emplace(op_type);
      }
    }
  }

  std::unordered_set<std::string> unsupported_ops;
  for (auto& op : all_ops) {
    if (!supported_ops.count(op)) {
      unsupported_ops.emplace(op);
    }
  }

  VLOG(4) << "-- The size of all_ops: " << all_ops.size() << " --";
  VLOG(4) << "-- The size of supported_ops: " << supported_ops.size() << " --";
  VLOG(4) << "-- The size of unsupported_ops: " << unsupported_ops.size()
          << " --";
  return std::make_tuple(std::move(all_ops), std::move(supported_ops),
                         std::move(unsupported_ops));
}

AmpOperators::AmpOperators()
    : allow_ops_(new std::unordered_set<std::string>()),
      block_ops_(new std::unordered_set<std::string>()),
      unsupported_fp16_ops_(new std::unordered_set<std::string>()),
      unsupported_bf16_ops_(new std::unordered_set<std::string>()) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  auto unsupported_ops_gpu_fp16 = std::get<2>(
      OpSupportedInfos("GPU", paddle::framework::proto::VarType::FP16));
  unsupported_fp16_ops_->insert(unsupported_ops_gpu_fp16.begin(),
                                unsupported_ops_gpu_fp16.end());
  auto unsupported_ops_gpu_bf16 = std::get<2>(
      OpSupportedInfos("GPU", paddle::framework::proto::VarType::BF16));
  unsupported_bf16_ops_->insert(unsupported_ops_gpu_bf16.begin(),
                                unsupported_ops_gpu_bf16.end());
// NOTE: GPU/NPU/XPU is compiled seperatly.
#elif defined(PADDLE_WITH_ASCEND_CL)
  auto unsupported_ops_npu_fp16 = std::get<2>(
      OpSupportedInfos("NPU", paddle::framework::proto::VarType::FP16));
  unsupported_fp16_ops_->insert(unsupported_ops_npu_fp16.begin(),
                                unsupported_ops_npu_fp16.end());
  auto unsupported_ops_npu_bf16 = std::get<2>(
      OpSupportedInfos("NPU", paddle::framework::proto::VarType::BF16));
  unsupported_bf16_ops_->insert(unsupported_ops_npu_bf16.begin(),
                                unsupported_ops_npu_bf16.end());
#elif defined(PADDLE_WITH_XPU)
  auto unsupported_ops_xpu_fp16 = std::get<2>(
      OpSupportedInfos("XPU", paddle::framework::proto::VarType::FP16));
  unsupported_fp16_ops_->insert(unsupported_ops_xpu_fp16.begin(),
                                unsupported_ops_xpu_fp16.end());
  auto unsupported_ops_xpu_bf16 = std::get<2>(
      OpSupportedInfos("XPU", paddle::framework::proto::VarType::BF16));
  unsupported_bf16_ops_->insert(unsupported_ops_xpu_bf16.begin(),
                                unsupported_ops_xpu_bf16.end());
#endif
  VLOG(4) << allow_ops_->size() << " " << block_ops_->size() << " "
          << unsupported_fp16_ops_->size() << " "
          << unsupported_bf16_ops_->size();
}

AmpOperators::~AmpOperators() {}

AmpOperators& AmpOperators::Instance() {
  static AmpOperators instance;
  return instance;
}

std::shared_ptr<std::unordered_set<std::string>>
AmpOperators::GetMutableAllowOps() {
  return allow_ops_;
}

std::shared_ptr<std::unordered_set<std::string>>
AmpOperators::GetMutableBlockOps() {
  return block_ops_;
}

std::shared_ptr<std::unordered_set<std::string>>
AmpOperators::GetMutableUnsupportedFp16Ops() {
  return unsupported_fp16_ops_;
}

std::shared_ptr<std::unordered_set<std::string>>
AmpOperators::GetMutableUnsupportedBf16Ops() {
  return unsupported_bf16_ops_;
}

std::ostream& operator<<(std::ostream& os, AmpOperators& ops) {
  os << "allow ops: ";
  auto allow_ops = ops.GetMutableAllowOps();
  std::copy((*allow_ops).begin(), (*allow_ops).end(),
            std::ostream_iterator<std::string>(os, " "));
  os << "\n";
  os << "block ops: ";
  auto block_ops = ops.GetMutableBlockOps();
  std::copy((*block_ops).begin(), (*block_ops).end(),
            std::ostream_iterator<std::string>(os, " "));
  os << "\n";
  os << "unsupported fp16 ops: ";
  auto unsupported_fp16_ops = ops.GetMutableUnsupportedFp16Ops();
  std::copy((*unsupported_fp16_ops).begin(), (*unsupported_fp16_ops).end(),
            std::ostream_iterator<std::string>(os, " "));
  os << "\n";
  os << "unsupported bf16 ops: ";
  auto unsupported_bf16_ops = ops.GetMutableUnsupportedBf16Ops();
  std::copy((*unsupported_bf16_ops).begin(), (*unsupported_bf16_ops).end(),
            std::ostream_iterator<std::string>(os, " "));
  return os;
}

}  // namespace imperative
}  // namespace paddle
