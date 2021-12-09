// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/eager/legacy/amp_auto_cast.h"
#include <memory>
#include <string>
#include "paddle/fluid/eager/legacy/op_runner.h"
#include "paddle/fluid/eager/legacy/tensor_helper.h"
#include "paddle/fluid/framework/operator.h"

namespace egr {
namespace legacy {

AmpOperators::AmpOperators()
    : allow_ops_(new std::unordered_set<std::string>()),
      block_ops_(new std::unordered_set<std::string>()),
      unsupported_fp16_ops_(new std::unordered_set<std::string>()) {
  auto& all_kernels = paddle::framework::OperatorWithKernel::AllOpKernels();
  auto fp16_dtype = paddle::framework::proto::VarType::FP16;
  for (auto it = all_kernels.begin(); it != all_kernels.end(); it++) {
    bool supported = false;
    for (auto& kernel_type : it->second) {
      if ((paddle::platform::is_gpu_place(kernel_type.first.place_) ||
           paddle::platform::is_xpu_place(kernel_type.first.place_)) &&
          kernel_type.first.data_type_ == fp16_dtype) {
        supported = true;
      }
    }
    if (!supported) {
      unsupported_fp16_ops_->insert(it->first);
    }
  }
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
  return os;
}

inline std::string GetDtypeStr(
    const std::shared_ptr<egr::EagerTensor>& tensor) {
  return paddle::framework::DataTypeToString(
      egr::legacy::GetDtypeFromVar(tensor->Var()));
}

inline bool NeedCast(const std::shared_ptr<egr::EagerTensor>& tensor) {
  auto place = egr::legacy::GetPlaceFromVar(tensor->Var());
  auto data_type = egr::legacy::GetDtypeFromVar(tensor->Var());
  if (paddle::platform::is_gpu_place(place) ||
      paddle::platform::is_cuda_pinned_place(place) ||
      paddle::platform::is_xpu_place(place)) {
    // CudaPinndePlace is added for varbase created by dataloader
    if (data_type == paddle::framework::proto::VarType::FP32 ||
        data_type == paddle::framework::proto::VarType::FP16) {
      return true;
    }
  }
  return false;
}

// NOTE: Trace a cast op, so if a var is casted from fp32 to fp16, then the grad
// var will be cast back from fp16 to fp32 during backward phase.
static inline std::shared_ptr<egr::EagerTensor> CastToType(
    const std::shared_ptr<egr::EagerTensor>& tensor,
    const paddle::framework::proto::VarType::Type dst_type) {
  NameTensorMap ins = {{"X", {tensor}}};
  auto in_data_type = egr::legacy::GetDtypeFromVar(tensor->Var());
  paddle::framework::AttributeMap attrs = {{"in_dtype", in_data_type},
                                           {"out_dtype", dst_type}};
  auto out = std::shared_ptr<egr::EagerTensor>(new egr::EagerTensor());
  NameTensorMap outs = {{"Out", {out}}};

  {
    AutoCastGuard guard(0);
    paddle::framework::AttributeMap default_attrs;
    RunOp("cast", ins, outs, std::move(attrs), {}, &default_attrs, true);
  }

  return out;
}

static inline std::shared_ptr<egr::EagerTensor> CastToFP16(
    const std::shared_ptr<egr::EagerTensor>& tensor) {
  auto dst_type = paddle::framework::proto::VarType::FP16;
  if (NeedCast(tensor) &&
      (egr::legacy::GetDtypeFromVar(tensor->Var()) != dst_type)) {
    return CastToType(tensor, dst_type);
  }
  return tensor;
}

static inline std::shared_ptr<egr::EagerTensor> CastToFP32(
    const std::shared_ptr<egr::EagerTensor>& tensor) {
  auto dst_type = paddle::framework::proto::VarType::FP32;
  if (NeedCast(tensor) &&
      (egr::legacy::GetDtypeFromVar(tensor->Var()) != dst_type)) {
    return CastToType(tensor, dst_type);
  }
  return tensor;
}

static inline paddle::framework::proto::VarType::Type GetPromoteType(
    const std::string& op_type, const NameTensorMap& ins) {
  auto dst_type = paddle::framework::proto::VarType::FP16;
  for (const auto& pair : ins) {
    for (const auto& tensor : pair.second) {
      if (egr::legacy::GetDtypeFromVar(tensor->Var()) ==
          paddle::framework::proto::VarType::FP32) {
        dst_type = egr::legacy::GetDtypeFromVar(tensor->Var());
        break;
      }
    }
  }

  // NOTE(juncai): moving_average_abs_max_scale only consider the
  // dtype of input(X)
  if (op_type == "moving_average_abs_max_scale") {
    for (const auto& pair : ins) {
      if (pair.first == "X" &&
          egr::legacy::GetDtypeFromVar(pair.second.front()->Var()) ==
              paddle::framework::proto::VarType::FP16) {
        dst_type = paddle::framework::proto::VarType::FP16;
      }
    }
  }

  return dst_type;
}

NameTensorMap AutoCastInputs(const std::string& op_type,
                             const NameTensorMap& ins) {
  NameTensorMap new_ins(ins);
  if (AmpOperators::Instance().GetMutableAllowOps()->count(op_type)) {
    for (auto& pair : new_ins) {
      // NOTE(zhiqiu): batch_norm and layer_norm support only input x is fp16.
      if ((op_type == "batch_norm" || op_type == "layer_norm" ||
           op_type == "sync_batch_norm") &&
          pair.first != "X") {
        continue;
      }

      VLOG(5) << "Op(" << op_type << "): Cast " << pair.first << " from "
              << GetDtypeStr(*pair.second.cbegin()) << " to float16";
      for (auto& var : pair.second) {
        var = CastToFP16(var);
      }
    }
    return new_ins;
  } else if (AmpOperators::Instance().GetMutableBlockOps()->count(op_type)) {
    for (auto& pair : new_ins) {
      VLOG(5) << "Op(" << op_type << "): Cast " << pair.first << " from "
              << GetDtypeStr(*pair.second.cbegin()) << " to float";
      for (auto& var : pair.second) {
        var = CastToFP32(var);
      }
    }
    return new_ins;
  } else {
    auto dst_type = GetPromoteType(op_type, ins);

    // NOTE(zhiqiu): if the op has op fp16 kernel, fall back to fp32.
    if (dst_type == paddle::framework::proto::VarType::FP16 &&
        AmpOperators::Instance().GetMutableUnsupportedFp16Ops()->count(
            op_type)) {
      dst_type = paddle::framework::proto::VarType::FP32;
    }
    for (auto& pair : new_ins) {
      // NOTE(zhiqiu): batch_norm and layer_norm support only input x is fp16.
      if ((op_type == "batch_norm" || op_type == "layer_norm" ||
           op_type == "sync_batch_norm") &&
          pair.first == "X" &&
          dst_type == paddle::framework::proto::VarType::FP32) {
        continue;
      }
      VLOG(5) << "Op(" << op_type << "): Cast " << pair.first << " from "
              << GetDtypeStr(*pair.second.cbegin()) << " to "
              << paddle::framework::DataTypeToString(dst_type);
      for (auto& var : pair.second) {
        var = (dst_type == paddle::framework::proto::VarType::FP32
                   ? CastToFP32(var)
                   : CastToFP16(var));
      }
    }
    return new_ins;
  }
  return new_ins;
}

NameTensorMap CastPureFp16Inputs(const std::string& op_type,
                                 const NameTensorMap& ins) {
  NameTensorMap new_ins(ins);
  auto dst_type = paddle::framework::proto::VarType::FP16;
  if (AmpOperators::Instance().GetMutableUnsupportedFp16Ops()->count(op_type) ||
      AmpOperators::Instance().GetMutableBlockOps()->count(op_type)) {
    dst_type = paddle::framework::proto::VarType::FP32;
  }
  for (auto& pair : new_ins) {
    if ((op_type == "batch_norm" || op_type == "layer_norm" ||
         op_type == "sync_batch_norm") &&
        pair.first != "X") {
      continue;
    }
    VLOG(5) << "Op(" << op_type << "): Cast " << pair.first << " from "
            << GetDtypeStr(*pair.second.cbegin()) << " to "
            << paddle::framework::DataTypeToString(dst_type);
    for (auto& var : pair.second) {
      var = (dst_type == paddle::framework::proto::VarType::FP32
                 ? CastToFP32(var)
                 : CastToFP16(var));
    }
  }
  return new_ins;
}

}  // namespace legacy
}  // namespace egr
