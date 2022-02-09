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

#include "paddle/fluid/imperative/amp_auto_cast.h"
#include <memory>
#include <string>
#include "paddle/fluid/eager/eager_tensor.h"
#include "paddle/fluid/imperative/tracer.h"
#include "paddle/fluid/imperative/type_defs.h"
#include "paddle/fluid/imperative/var_helper.h"

namespace paddle {
namespace imperative {

class VarBase;

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

  auto pten_kernels = pten::KernelFactory::Instance().kernels();
  for (auto& kernel_pair : pten_kernels) {
    auto op_type = pten::TransToFluidOpName(kernel_pair.first);
    for (auto& info_pair : kernel_pair.second) {
      framework::OpKernelType kernel_type =
          framework::TransPtenKernelKeyToOpKernelType(info_pair.first);
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

AutoCastGuard::AutoCastGuard(std::shared_ptr<Tracer> tracer, AmpLevel level)
    : tracer_(tracer) {
  pre_amp_level_ = tracer_->GetAmpLevel();

  if (pre_amp_level_ != level) {
    tracer_->SetAmpLevel(level);
  }
}

AutoCastGuard::~AutoCastGuard() { tracer_->SetAmpLevel(pre_amp_level_); }

AmpOperators::AmpOperators()
    : allow_ops_(new std::unordered_set<std::string>()),
      block_ops_(new std::unordered_set<std::string>()),
      unsupported_fp16_ops_(new std::unordered_set<std::string>()) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  auto unsupported_ops_gpu = std::get<2>(
      OpSupportedInfos("GPU", paddle::framework::proto::VarType::FP16));
  unsupported_fp16_ops_->insert(unsupported_ops_gpu.begin(),
                                unsupported_ops_gpu.end());
// NOTE: GPU/NPU/XPU is compiled seperatly.
#elif defined(PADDLE_WITH_ASCEND_CL)
  auto unsupported_ops_npu = std::get<2>(
      OpSupportedInfos("NPU", paddle::framework::proto::VarType::FP16));
  unsupported_fp16_ops_->insert(unsupported_ops_npu.begin(),
                                unsupported_ops_npu.end());
#elif defined(PADDLE_WITH_XPU)
  auto unsupported_ops_xpu = std::get<2>(
      OpSupportedInfos("XPU", paddle::framework::proto::VarType::FP16));
  unsupported_fp16_ops_->insert(unsupported_ops_xpu.begin(),
                                unsupported_ops_xpu.end());
#endif
  VLOG(4) << allow_ops_->size() << " " << block_ops_->size() << " "
          << unsupported_fp16_ops_->size();
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

template <typename VarType>
inline std::string GetDtypeStr(const std::shared_ptr<VarType>& var) {
  return framework::DataTypeToString(GetDataType<VarType>(var));
}
template <typename VarType>
inline bool NeedCast(const std::shared_ptr<VarType>& var) {
  auto place = GetPlace(var);
  auto data_type = GetDataType<VarType>(var);
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
template <typename VarType>
static inline std::shared_ptr<VarType> CastToType(
    const std::shared_ptr<VarType>& var,
    const framework::proto::VarType::Type dst_type) {
  const auto& tracer = imperative::GetCurrentTracer();
  imperative::NameVarMap<VarType> ins = {{"X", {var}}};
  framework::AttributeMap attrs = {{"in_dtype", GetDataType<VarType>(var)},
                                   {"out_dtype", dst_type}};
  auto out =
      std::shared_ptr<VarType>(new VarType(tracer->GenerateUniqueName()));
  imperative::NameVarMap<VarType> outs = {{"Out", {out}}};

  {
    AutoCastGuard guard(tracer, AmpLevel::O0);
    tracer->TraceOp("cast", ins, outs, std::move(attrs));
  }

  return out;
}
template <typename VarType>
static inline std::shared_ptr<VarType> CastToFP16(
    const std::shared_ptr<VarType>& var) {
  auto dst_type = framework::proto::VarType::FP16;
  if (NeedCast(var) && (GetDataType<VarType>(var) != dst_type)) {
    return CastToType(var, dst_type);
  }
  return var;
}

template <typename VarType>
static inline std::shared_ptr<VarType> CastToFP32(
    const std::shared_ptr<VarType>& var) {
  auto dst_type = framework::proto::VarType::FP32;
  if (NeedCast(var) && (GetDataType<VarType>(var) != dst_type)) {
    return CastToType(var, dst_type);
  }
  return var;
}

template <typename VarType>
static inline framework::proto::VarType::Type GetPromoteType(
    const std::string& op_type, const NameVarMap<VarType>& ins) {
  auto dst_type = framework::proto::VarType::FP16;
  for (const auto& pair : ins) {
    for (const auto& var : pair.second) {
      if (GetDataType<VarType>(var) == framework::proto::VarType::FP32) {
        dst_type = GetDataType<VarType>(var);
        break;
      }
    }
  }

  // NOTE(juncai): moving_average_abs_max_scale only consider the
  // dtype of input(X)
  if (op_type == "moving_average_abs_max_scale") {
    for (const auto& pair : ins) {
      if (pair.first == "X" &&
          GetDataType<VarType>(pair.second.front()) ==
              framework::proto::VarType::FP16) {
        dst_type = framework::proto::VarType::FP16;
      }
    }
  }

  return dst_type;
}

template <typename VarType>
NameVarMap<VarType> AutoCastInputs(const std::string& op_type,
                                   const NameVarMap<VarType>& ins) {
  NameVarMap<VarType> new_ins(ins);
  if (AmpOperators::Instance().GetMutableAllowOps()->count(op_type)) {
    for (auto& pair : new_ins) {
      // NOTE(zhiqiu): batch_norm and layer_norm support only input x is fp16.
      if ((op_type == "batch_norm" || op_type == "layer_norm" ||
           op_type == "sync_batch_norm") &&
          pair.first != "X") {
        continue;
      }

      if ((op_type == "fused_attention" || op_type == "fused_feedforward")) {
        if (pair.first == "LnScale" || pair.first == "LnBias" ||
            pair.first == "Ln2Scale" || pair.first == "Ln2Bias" ||
            pair.first == "Ln1Scale" || pair.first == "Ln1Bias") {
          continue;
        }
      }

      VLOG(5) << "Op(" << op_type << "): Cast " << pair.first << " from "
              << GetDtypeStr(*pair.second.cbegin()) << " to float16";
      for (auto& var : pair.second) {
        var = CastToFP16<VarType>(var);
      }
    }
    return new_ins;
  } else if (AmpOperators::Instance().GetMutableBlockOps()->count(op_type)) {
    for (auto& pair : new_ins) {
      VLOG(5) << "Op(" << op_type << "): Cast " << pair.first << " from "
              << GetDtypeStr(*pair.second.cbegin()) << " to float";
      for (auto& var : pair.second) {
        var = CastToFP32<VarType>(var);
      }
    }
    return new_ins;
  } else {
    auto dst_type = GetPromoteType<VarType>(op_type, ins);

    // NOTE(zhiqiu): if the op has op fp16 kernel, fall back to fp32.
    if (dst_type == framework::proto::VarType::FP16 &&
        AmpOperators::Instance().GetMutableUnsupportedFp16Ops()->count(
            op_type)) {
      dst_type = framework::proto::VarType::FP32;
    }
    for (auto& pair : new_ins) {
      // NOTE(zhiqiu): batch_norm and layer_norm support only input x is fp16.
      if ((op_type == "batch_norm" || op_type == "layer_norm" ||
           op_type == "sync_batch_norm") &&
          pair.first == "X" && dst_type == framework::proto::VarType::FP32) {
        continue;
      }
      if ((op_type == "fused_attention" || op_type == "fused_feedforwad") &&
          dst_type == framework::proto::VarType::FP32) {
        if (pair.first != "LnScale" && pair.first != "LnBias" &&
            pair.first != "Ln2Scale" && pair.first != "Ln2Bias" &&
            pair.first != "Ln1Scale" && pair.first != "Ln1Bias") {
          continue;
        }
      }
      VLOG(5) << "Op(" << op_type << "): Cast " << pair.first << " from "
              << GetDtypeStr(*pair.second.cbegin()) << " to "
              << framework::DataTypeToString(dst_type);
      for (auto& var : pair.second) {
        var = (dst_type == framework::proto::VarType::FP32
                   ? CastToFP32<VarType>(var)
                   : CastToFP16<VarType>(var));
      }
    }
    return new_ins;
  }
  return new_ins;
}
template NameVarMap<VarBase> AutoCastInputs<VarBase>(
    const std::string& op_type, const NameVarMap<VarBase>& ins);
template NameVarMap<egr::EagerTensor> AutoCastInputs<egr::EagerTensor>(
    const std::string& op_type, const NameVarMap<egr::EagerTensor>& ins);
template <typename VarType>
NameVarMap<VarType> CastPureFp16Inputs(const std::string& op_type,
                                       const NameVarMap<VarType>& ins) {
  NameVarMap<VarType> new_ins(ins);
  auto dst_type = framework::proto::VarType::FP16;
  if (AmpOperators::Instance().GetMutableUnsupportedFp16Ops()->count(op_type) ||
      AmpOperators::Instance().GetMutableBlockOps()->count(op_type)) {
    dst_type = framework::proto::VarType::FP32;
  }
  for (auto& pair : new_ins) {
    // NOTE: The run_program OP only has FP32 kernel. In dy2stat pure fp16
    // training, we have correctly cast the inputs of run_program OP before,
    // so here should avoid casting for run_program OP.
    if (op_type == "run_program") {
      continue;
    }

    if ((op_type == "batch_norm" || op_type == "layer_norm" ||
         op_type == "sync_batch_norm") &&
        pair.first != "X") {
      continue;
    }
    if ((op_type == "fused_attention" || op_type == "fused_feedforward")) {
      if (pair.first == "LnScale" || pair.first == "LnBias" ||
          pair.first == "Ln2Scale" || pair.first == "Ln2Bias" ||
          pair.first == "Ln1Scale" || pair.first == "Ln1Bias") {
        continue;
      }
    }
    VLOG(5) << "Op(" << op_type << "): Cast " << pair.first << " from "
            << GetDtypeStr(*pair.second.cbegin()) << " to "
            << framework::DataTypeToString(dst_type);
    for (auto& var : pair.second) {
      var = (dst_type == framework::proto::VarType::FP32
                 ? CastToFP32<VarType>(var)
                 : CastToFP16<VarType>(var));
    }
  }
  return new_ins;
}
template NameVarMap<VarBase> CastPureFp16Inputs<VarBase>(
    const std::string& op_type, const NameVarMap<VarBase>& ins);
template NameVarMap<egr::EagerTensor> CastPureFp16Inputs<egr::EagerTensor>(
    const std::string& op_type, const NameVarMap<egr::EagerTensor>& ins);
}  // namespace imperative
}  // namespace paddle
