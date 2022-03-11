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
#include "paddle/fluid/imperative/type_defs.h"
#include "paddle/fluid/imperative/amp_utils.h"
#include "paddle/fluid/eager/eager_tensor.h"
#include "paddle/fluid/imperative/tracer.h"
#include "paddle/fluid/imperative/var_helper.h"
#include "paddle/fluid/eager/api/generated/fluid_generated/dygraph_forward_api.h"
#include "paddle/fluid/eager/api/utils/global_utils.h"
#include "paddle/fluid/imperative/layer.h"

namespace egr {

// class Tracer;

// class paddle::imperative::VarBase;

// NOTE(zhiqiu): AutoCastGuard is used for RAII.
class EagerAutoCastGuard {
 public:
  EagerAutoCastGuard(std::shared_ptr<paddle::imperative::Tracer> tracer, paddle::imperative::AmpLevel guard_level)
      : tracer_(tracer) {
    pre_amp_level_ = tracer_->GetAmpLevel();

    if (pre_amp_level_ != guard_level) {
      tracer_->SetAmpLevel(guard_level);
    }
  }

  ~EagerAutoCastGuard() { tracer_->SetAmpLevel(pre_amp_level_); }

  // forbid copy and operator=
  EagerAutoCastGuard(const EagerAutoCastGuard& guard) = delete;
  EagerAutoCastGuard& operator=(const EagerAutoCastGuard& guard) = delete;

 private:
  std::shared_ptr<paddle::imperative::Tracer> tracer_;
  paddle::imperative::AmpLevel pre_amp_level_;
};

template <typename VarType>
inline std::string EagerGetDtypeStr(const std::shared_ptr<VarType>& var) {
  return paddle::framework::DataTypeToString(paddle::imperative::GetDataType<VarType>(var));
}
template <typename VarType>
inline bool EagerNeedCast(const std::shared_ptr<VarType>& var) {
  auto place = paddle::imperative::GetPlace(var);
  auto data_type = paddle::imperative::GetDataType<VarType>(var);
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
static inline std::shared_ptr<VarType> EagerCastToType(
    const std::shared_ptr<VarType>& var,
    const paddle::framework::proto::VarType::Type dst_type) {
  const auto& tracer = paddle::imperative::GetCurrentTracer();
  paddle::imperative::NameVarMap<VarType> ins = {{"X", {var}}};
  paddle::framework::AttributeMap attrs = {{"in_dtype", paddle::imperative::GetDataType<VarType>(var)},
                                   {"out_dtype", dst_type}};
  auto out =
      std::shared_ptr<VarType>(new VarType(tracer->GenerateUniqueName()));
  paddle::imperative::NameVarMap<VarType> outs = {{"Out", {out}}};

  {
    EagerAutoCastGuard guard(tracer, paddle::imperative::AmpLevel::O0);
    tracer->TraceOp("cast", ins, outs, std::move(attrs));
  }

  return out;
}
template <>
std::shared_ptr<egr::EagerVariable> EagerCastToType<egr::EagerVariable>(
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
    EagerAutoCastGuard guard(tracer, paddle::imperative::AmpLevel::O0);
    VLOG(6) << " EagerMode: " << egr::Controller::Instance().InEagerMode(); 
    if (egr::Controller::Instance().InEagerMode()){
      VLOG(6) << " use EagerMode cast "; 
      paddle::experimental::Tensor input;
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
static inline std::shared_ptr<VarType> EagerCastToFP16(
    const std::shared_ptr<VarType>& var) {
  auto dst_type = paddle::framework::proto::VarType::FP16;
  if (EagerNeedCast(var) && (paddle::imperative::GetDataType<VarType>(var) != dst_type)) {
    return EagerCastToType(var, dst_type);
  }
  return var;
}

template <typename VarType>
static inline std::shared_ptr<VarType> EagerCastToFP32(
    const std::shared_ptr<VarType>& var) {
  auto dst_type = paddle::framework::proto::VarType::FP32;
  if (EagerNeedCast(var) && (paddle::imperative::GetDataType<VarType>(var) != dst_type)) {
    return EagerCastToType(var, dst_type);
  }
  return var;
}

template <typename VarType>
static inline std::shared_ptr<VarType> EagerCastToBF16(
    const std::shared_ptr<VarType>& var) {
  auto dst_type = paddle::framework::proto::VarType::BF16;
  if (EagerNeedCast(var) && (paddle::imperative::GetDataType<VarType>(var) != dst_type)) {
    return EagerCastToType(var, dst_type);
  }
  return var;
}

template <typename VarType>
static inline paddle::framework::proto::VarType::Type EagerGetPromoteType(
    const std::string& op_type, const paddle::imperative::NameVarMap<VarType>& ins,
    const paddle::framework::proto::VarType::Type amp_dtype) {
  auto dst_type = amp_dtype;
  for (const auto& pair : ins) {
    for (const auto& var : pair.second) {
      if (paddle::imperative::GetDataType<VarType>(var) == paddle::framework::proto::VarType::FP32) {
        dst_type = paddle::imperative::GetDataType<VarType>(var);
        break;
      }
    }
  }

  // NOTE(juncai): moving_average_abs_max_scale only consider the
  // dtype of input(X)
  if (op_type == "moving_average_abs_max_scale") {
    for (const auto& pair : ins) {
      if (pair.first == "X" &&
          paddle::imperative::GetDataType<VarType>(pair.second.front()) ==
              paddle::framework::proto::VarType::FP16) {
        dst_type = paddle::framework::proto::VarType::FP16;
      }
    }
  }

  return dst_type;
}


template <typename VarType>
paddle::imperative::NameVarMap<VarType> EagerAutoCastInputs(const std::string& op_type,
                                   const paddle::imperative::NameVarMap<VarType>& ins) {
  paddle::imperative::NameVarMap<VarType> new_ins(ins);
  if (paddle::imperative::AmpOperators::Instance().GetMutableAllowOps()->count(op_type)) {
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
              << EagerGetDtypeStr(*pair.second.cbegin()) << " to float16";
      for (auto& var : pair.second) {
        var = EagerCastToFP16<VarType>(var);
      }
    }
    return new_ins;
  } else if (paddle::imperative::AmpOperators::Instance().GetMutableBlockOps()->count(op_type)) {
    for (auto& pair : new_ins) {
      VLOG(5) << "Op(" << op_type << "): Cast " << pair.first << " from "
              << EagerGetDtypeStr(*pair.second.cbegin()) << " to float";
      for (auto& var : pair.second) {
        var = EagerCastToFP32<VarType>(var);
      }
    }
    return new_ins;
  } else {
    auto dst_type =
        EagerGetPromoteType<VarType>(op_type, ins, paddle::framework::proto::VarType::FP16);

    // NOTE(zhiqiu): if the op has op fp16 kernel, fall back to fp32.
    if (dst_type == paddle::framework::proto::VarType::FP16 &&
        paddle::imperative::AmpOperators::Instance().GetMutableUnsupportedFp16Ops()->count(
            op_type)) {
      dst_type = paddle::framework::proto::VarType::FP32;
    }
    for (auto& pair : new_ins) {
      // NOTE(zhiqiu): batch_norm and layer_norm support only input x is fp16.
      if ((op_type == "batch_norm" || op_type == "layer_norm" ||
           op_type == "sync_batch_norm") &&
          pair.first == "X" && dst_type == paddle::framework::proto::VarType::FP32) {
        continue;
      }
      if ((op_type == "fused_attention" || op_type == "fused_feedforwad") &&
          dst_type == paddle::framework::proto::VarType::FP32) {
        if (pair.first != "LnScale" && pair.first != "LnBias" &&
            pair.first != "Ln2Scale" && pair.first != "Ln2Bias" &&
            pair.first != "Ln1Scale" && pair.first != "Ln1Bias") {
          continue;
        }
      }
      VLOG(5) << "Op(" << op_type << "): Cast " << pair.first << " from "
              << EagerGetDtypeStr(*pair.second.cbegin()) << " to "
              << paddle::framework::DataTypeToString(dst_type);
      for (auto& var : pair.second) {
        var = (dst_type == paddle::framework::proto::VarType::FP32
                   ? EagerCastToFP32<VarType>(var)
                   : EagerCastToFP16<VarType>(var));
      }
    }
    return new_ins;
  }
  return new_ins;
}
template paddle::imperative::NameVarMap<paddle::imperative::VarBase> EagerAutoCastInputs<paddle::imperative::VarBase>(
    const std::string& op_type, const paddle::imperative::NameVarMap<paddle::imperative::VarBase>& ins);
template paddle::imperative::NameVarMap<egr::EagerVariable> EagerAutoCastInputs<egr::EagerVariable>(
    const std::string& op_type, const paddle::imperative::NameVarMap<egr::EagerVariable>& ins);

template <typename VarType>
paddle::imperative::NameVarMap<VarType> EagerCastPureFp16Inputs(const std::string& op_type,
                                       const paddle::imperative::NameVarMap<VarType>& ins) {
  paddle::imperative::NameVarMap<VarType> new_ins(ins);
  auto dst_type = paddle::framework::proto::VarType::FP16;
  if (paddle::imperative::AmpOperators::Instance().GetMutableUnsupportedFp16Ops()->count(op_type) ||
      paddle::imperative::AmpOperators::Instance().GetMutableBlockOps()->count(op_type)) {
    dst_type = paddle::framework::proto::VarType::FP32;
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
            << EagerGetDtypeStr(*pair.second.cbegin()) << " to "
            << paddle::framework::DataTypeToString(dst_type);
    for (auto& var : pair.second) {
      var = (dst_type == paddle::framework::proto::VarType::FP32
                 ? EagerCastToFP32<VarType>(var)
                 : EagerCastToFP16<VarType>(var));
    }
  }
  return new_ins;
}
template paddle::imperative::NameVarMap<paddle::imperative::VarBase> EagerCastPureFp16Inputs<paddle::imperative::VarBase>(
    const std::string& op_type, const paddle::imperative::NameVarMap<paddle::imperative::VarBase>& ins);
template paddle::imperative::NameVarMap<egr::EagerVariable> EagerCastPureFp16Inputs<egr::EagerVariable>(
    const std::string& op_type, const paddle::imperative::NameVarMap<egr::EagerVariable>& ins);

template <typename VarType>
paddle::imperative::NameVarMap<VarType> EagerAutoCastBF16Inputs(const std::string& op_type,
                                       const paddle::imperative::NameVarMap<VarType>& ins) {
  paddle::imperative::NameVarMap<VarType> new_ins(ins);
  if (paddle::imperative::AmpOperators::Instance().GetMutableAllowOps()->count(op_type)) {
    for (auto& pair : new_ins) {
      VLOG(5) << "Op(" << op_type << "): Cast " << pair.first << " from "
              << EagerGetDtypeStr(*pair.second.cbegin()) << " to bfloat16";
      for (auto& var : pair.second) {
        var = EagerCastToBF16<VarType>(var);
      }
    }
    return new_ins;
  } else if (paddle::imperative::AmpOperators::Instance().GetMutableBlockOps()->count(op_type)) {
    for (auto& pair : new_ins) {
      VLOG(5) << "Op(" << op_type << "): Cast " << pair.first << " from "
              << EagerGetDtypeStr(*pair.second.cbegin()) << " to float";
      for (auto& var : pair.second) {
        var = EagerCastToFP32<VarType>(var);
      }
    }
    return new_ins;
  } else {
    auto dst_type =
        EagerGetPromoteType<VarType>(op_type, ins, paddle::framework::proto::VarType::BF16);
    // NOTE(zhangbo): if the op has op fp16 kernel, fall back to fp32.
    if (dst_type == paddle::framework::proto::VarType::BF16 &&
        paddle::imperative::AmpOperators::Instance().GetMutableUnsupportedBf16Ops()->count(
            op_type)) {
      dst_type = paddle::framework::proto::VarType::FP32;
    }
    for (auto& pair : new_ins) {
      VLOG(5) << "Op(" << op_type << "): Cast " << pair.first << " from "
              << EagerGetDtypeStr(*pair.second.cbegin()) << " to "
              << paddle::framework::DataTypeToString(dst_type);
      for (auto& var : pair.second) {
        var = (dst_type == paddle::framework::proto::VarType::FP32
                   ? EagerCastToFP32<VarType>(var)
                   : EagerCastToBF16<VarType>(var));
      }
    }
    return new_ins;
  }
  return new_ins;
}
template paddle::imperative::NameVarMap<paddle::imperative::VarBase> EagerAutoCastBF16Inputs<paddle::imperative::VarBase>(
    const std::string& op_type, const paddle::imperative::NameVarMap<paddle::imperative::VarBase>& ins);
template paddle::imperative::NameVarMap<egr::EagerVariable> EagerAutoCastBF16Inputs<egr::EagerVariable>(
    const std::string& op_type, const paddle::imperative::NameVarMap<egr::EagerVariable>& ins);

template <typename VarType>
paddle::imperative::NameVarMap<VarType> EagerCastPureBf16Inputs(const std::string& op_type,
                                       const paddle::imperative::NameVarMap<VarType>& ins) {
  paddle::imperative::NameVarMap<VarType> new_ins(ins);
  auto dst_type = paddle::framework::proto::VarType::BF16;
  if (paddle::imperative::AmpOperators::Instance().GetMutableUnsupportedBf16Ops()->count(op_type) ||
      paddle::imperative::AmpOperators::Instance().GetMutableBlockOps()->count(op_type)) {
    dst_type = paddle::framework::proto::VarType::FP32;
  }
  for (auto& pair : new_ins) {
    VLOG(5) << "Op(" << op_type << "): Cast " << pair.first << " from "
            << EagerGetDtypeStr(*pair.second.cbegin()) << " to "
            << paddle::framework::DataTypeToString(dst_type);
    for (auto& var : pair.second) {
      var = (dst_type == paddle::framework::proto::VarType::FP32
                 ? EagerCastToFP32<VarType>(var)
                 : EagerCastToBF16<VarType>(var));
    }
  }
  return new_ins;
}
template paddle::imperative::NameVarMap<paddle::imperative::VarBase> EagerCastPureBf16Inputs<paddle::imperative::VarBase>(
    const std::string& op_type, const paddle::imperative::NameVarMap<paddle::imperative::VarBase>& ins);
template paddle::imperative::NameVarMap<egr::EagerVariable> EagerCastPureBf16Inputs<egr::EagerVariable>(
    const std::string& op_type, const paddle::imperative::NameVarMap<egr::EagerVariable>& ins);

}  // namespace egr
