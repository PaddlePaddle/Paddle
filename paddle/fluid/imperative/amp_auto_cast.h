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

#pragma once
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <unordered_set>

// #include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/imperative/type_defs.h"
#include "paddle/fluid/imperative/amp_utils.h"
#include "paddle/fluid/eager/eager_tensor.h"
#include "paddle/fluid/imperative/tracer.h"
#include "paddle/fluid/imperative/var_helper.h"
#include "paddle/fluid/eager/api/generated/fluid_generated/dygraph_forward_api.h"
#include "paddle/fluid/eager/api/utils/global_utils.h"

namespace paddle {
namespace imperative {

// // NOTE(zhiqiu): only O1 and O2 are valid now
// enum class AmpLevel {
//   O0 = 0,  // fp32
//   O1,      // amp, mixed fp32-fp16
//   O2,      // almost fp16
//   O3,      // fp16
// };

// std::tuple<std::unordered_set<std::string>, std::unordered_set<std::string>,
//            std::unordered_set<std::string>>
// OpSupportedInfos(const std::string& place,
//                  framework::proto::VarType::Type dtype);

class Tracer;

class VarBase;

// // Singleton implementation with C++ 11
// class AmpOperators {
//  public:
//   ~AmpOperators();
//   AmpOperators(const AmpOperators& o) = delete;
//   const AmpOperators& operator=(const AmpOperators& o) = delete;

//   static AmpOperators& Instance();

//   std::shared_ptr<std::unordered_set<std::string>> GetMutableAllowOps();

//   std::shared_ptr<std::unordered_set<std::string>> GetMutableBlockOps();

//   std::shared_ptr<std::unordered_set<std::string>>
//   GetMutableUnsupportedFp16Ops();

//   std::shared_ptr<std::unordered_set<std::string>>
//   GetMutableUnsupportedBf16Ops();

//  private:
//   AmpOperators();  // forbid calling default constructor

//   // The set of ops that support fp16 calculation and are considered numerically
//   // safe and performance critical. These ops are always converted to fp16.
//   std::shared_ptr<std::unordered_set<std::string>> allow_ops_;

//   // The set of ops that support fp16 calculation and are considered numerically
//   // dangerous and whose effects may also be observed in downstream ops.
//   std::shared_ptr<std::unordered_set<std::string>> block_ops_;

//   // The set of ops that has no fp16 CUDA kennel.
//   std::shared_ptr<std::unordered_set<std::string>> unsupported_fp16_ops_;

//   // The set of ops that has no bf16 CUDA kennel.
//   std::shared_ptr<std::unordered_set<std::string>> unsupported_bf16_ops_;
// };

// std::ostream& operator<<(std::ostream& os, AmpOperators& ops);

// NOTE(zhiqiu): AutoCastGuard is used for RAII.
class AutoCastGuard {
 public:
  AutoCastGuard(std::shared_ptr<Tracer> tracer, AmpLevel guard_level)
      : tracer_(tracer) {
    pre_amp_level_ = tracer_->GetAmpLevel();

    if (pre_amp_level_ != guard_level) {
      tracer_->SetAmpLevel(guard_level);
    }
  }

  ~AutoCastGuard() { tracer_->SetAmpLevel(pre_amp_level_); }

  // forbid copy and operator=
  AutoCastGuard(const AutoCastGuard& guard) = delete;
  AutoCastGuard& operator=(const AutoCastGuard& guard) = delete;

 private:
  std::shared_ptr<Tracer> tracer_;
  AmpLevel pre_amp_level_;
};

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
        data_type == paddle::framework::proto::VarType::FP16 ||
        data_type == paddle::framework::proto::VarType::BF16) {
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

template <>
std::shared_ptr<egr::EagerVariable> CastToType<egr::EagerVariable>(
    const std::shared_ptr<egr::EagerVariable>& var,
    const paddle::framework::proto::VarType::Type dst_type){
  const auto& tracer = paddle::imperative::GetCurrentTracer();
  paddle::imperative::NameVarMap<egr::EagerVariable> ins = {{"X", {var}}};
  paddle::framework::AttributeMap attrs = {{"in_dtype", paddle::imperative::GetDataType<egr::EagerVariable>(var)},
                                   {"out_dtype", dst_type}};
  auto out =
      std::shared_ptr<egr::EagerVariable>(new egr::EagerVariable(tracer->GenerateUniqueName()));
  paddle::imperative::NameVarMap<egr::EagerVariable> outs = {{"Out", {out}}};

  {
    AutoCastGuard guard(tracer, paddle::imperative::AmpLevel::O0);
    VLOG(6) << " EagerMode: " << egr::Controller::Instance().InEagerMode(); 
    VLOG(6) << " EagerMode grad state: " << egr::Controller::Instance().HasGrad(); 
    if (egr::Controller::Instance().InEagerMode()){
      VLOG(6) << " use EagerMode cast "; 
      paddle::experimental::Tensor input;
      VLOG(6) << " input auto_grad_meta " << egr::EagerUtils::nullable_autograd_meta(input);
      egr::EagerUtils::GetOutput(var, &input);
      auto out_temp = cast_dygraph_function(input, std::move(attrs));
      out = egr::EagerUtils::TrySyncToVar(out_temp);
    } else {
      tracer->TraceOp("cast", ins, outs, std::move(attrs));
    }
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
static inline std::shared_ptr<VarType> CastToBF16(
    const std::shared_ptr<VarType>& var) {
  auto dst_type = framework::proto::VarType::BF16;
  if (NeedCast(var) && (GetDataType<VarType>(var) != dst_type)) {
    return CastToType(var, dst_type);
  }
  return var;
}

template <typename VarType>
static inline framework::proto::VarType::Type GetPromoteType(
    const std::string& op_type, const NameVarMap<VarType>& ins,
    const framework::proto::VarType::Type amp_dtype) {
  auto dst_type = amp_dtype;
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

// template <typename VarType>
// NameVarMap<VarType> AutoCastInputs(const std::string& op_type,
//                                    const NameVarMap<VarType>& ins);
// template <typename VarType>
// NameVarMap<VarType> CastPureFp16Inputs(const std::string& op_type,
//                                        const NameVarMap<VarType>& ins);
// template <typename VarType>
// NameVarMap<VarType> AutoCastBF16Inputs(const std::string& op_type,
//                                        const NameVarMap<VarType>& ins);
// template <typename VarType>
// NameVarMap<VarType> CastPureBf16Inputs(const std::string& op_type,
//                                        const NameVarMap<VarType>& ins);


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
    auto dst_type =
        GetPromoteType<VarType>(op_type, ins, framework::proto::VarType::FP16);

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
template NameVarMap<egr::EagerVariable> AutoCastInputs<egr::EagerVariable>(
    const std::string& op_type, const NameVarMap<egr::EagerVariable>& ins);
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
template NameVarMap<egr::EagerVariable> CastPureFp16Inputs<egr::EagerVariable>(
    const std::string& op_type, const NameVarMap<egr::EagerVariable>& ins);

template <typename VarType>
NameVarMap<VarType> AutoCastBF16Inputs(const std::string& op_type,
                                       const NameVarMap<VarType>& ins) {
  NameVarMap<VarType> new_ins(ins);
  if (AmpOperators::Instance().GetMutableAllowOps()->count(op_type)) {
    for (auto& pair : new_ins) {
      VLOG(5) << "Op(" << op_type << "): Cast " << pair.first << " from "
              << GetDtypeStr(*pair.second.cbegin()) << " to bfloat16";
      for (auto& var : pair.second) {
        var = CastToBF16<VarType>(var);
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
    auto dst_type =
        GetPromoteType<VarType>(op_type, ins, framework::proto::VarType::BF16);
    // NOTE(zhangbo): if the op has op fp16 kernel, fall back to fp32.
    if (dst_type == framework::proto::VarType::BF16 &&
        AmpOperators::Instance().GetMutableUnsupportedBf16Ops()->count(
            op_type)) {
      dst_type = framework::proto::VarType::FP32;
    }
    for (auto& pair : new_ins) {
      VLOG(5) << "Op(" << op_type << "): Cast " << pair.first << " from "
              << GetDtypeStr(*pair.second.cbegin()) << " to "
              << framework::DataTypeToString(dst_type);
      for (auto& var : pair.second) {
        var = (dst_type == framework::proto::VarType::FP32
                   ? CastToFP32<VarType>(var)
                   : CastToBF16<VarType>(var));
      }
    }
    return new_ins;
  }
  return new_ins;
}
template NameVarMap<VarBase> AutoCastBF16Inputs<VarBase>(
    const std::string& op_type, const NameVarMap<VarBase>& ins);
template NameVarMap<egr::EagerVariable> AutoCastBF16Inputs<egr::EagerVariable>(
    const std::string& op_type, const NameVarMap<egr::EagerVariable>& ins);

template <typename VarType>
NameVarMap<VarType> CastPureBf16Inputs(const std::string& op_type,
                                       const NameVarMap<VarType>& ins) {
  NameVarMap<VarType> new_ins(ins);
  auto dst_type = framework::proto::VarType::BF16;
  if (AmpOperators::Instance().GetMutableUnsupportedBf16Ops()->count(op_type) ||
      AmpOperators::Instance().GetMutableBlockOps()->count(op_type)) {
    dst_type = framework::proto::VarType::FP32;
  }
  for (auto& pair : new_ins) {
    VLOG(5) << "Op(" << op_type << "): Cast " << pair.first << " from "
            << GetDtypeStr(*pair.second.cbegin()) << " to "
            << framework::DataTypeToString(dst_type);
    for (auto& var : pair.second) {
      var = (dst_type == framework::proto::VarType::FP32
                 ? CastToFP32<VarType>(var)
                 : CastToBF16<VarType>(var));
    }
  }
  return new_ins;
}
template NameVarMap<VarBase> CastPureBf16Inputs<VarBase>(
    const std::string& op_type, const NameVarMap<VarBase>& ins);
template NameVarMap<egr::EagerVariable> CastPureBf16Inputs<egr::EagerVariable>(
    const std::string& op_type, const NameVarMap<egr::EagerVariable>& ins);

}  // namespace imperative
}  // namespace paddle
