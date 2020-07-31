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

#include <algorithm>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>

#include "paddle/fluid/imperative/layer.h"
#include "paddle/fluid/imperative/tracer.h"
#include "paddle/fluid/imperative/variable_wrapper.h"

namespace paddle {
namespace imperative {

// TODO(zhiqiu): consider to cache casted params for better inference
// performance.

// The set of ops that support fp16 calculation and are considered numerically -
// safe and performance critical. These ops are always converted to fp16.
static std::unordered_set<std::string> g_white_ops;

// The set of ops that support fp16 calculation and are considered numerically -
// dangerous and whose effects may also be observed in downstream ops.
static std::unordered_set<std::string> g_black_ops;

// This set contains two types of ops. All ops supported fp16 calculation. One
// of two types is considered numerically - safe, but may be made unsafe by an
// upstream blacklist op. Another type do not have numerically - significant
// effects, like stack, flatten2.
// static std::unordered_set<std::string> g_gray_ops;

inline std::string GetDtypeStr(
    const std::shared_ptr<imperative::VarBase>& var) {
  return framework::DataTypeToString(var->DataType());
}

inline bool NeedCast(const std::shared_ptr<VarBase>& var) {
  if (!platform::is_gpu_place(var->Place())) {
    return false;
  }
  if (var->DataType() == framework::proto::VarType::FP32 ||
      var->DataType() == framework::proto::VarType::FP16) {
    return true;
  } else {
    return false;
  }
}

// NOTE: Trace a cast op, so if a var is casted from fp32 to fp16, then the grad
// var will be cast back from fp16 to fp32 during backward phase.
static inline std::shared_ptr<imperative::VarBase> CastToType(
    const std::shared_ptr<VarBase>& var,
    const framework::proto::VarType::Type dst_type) {
  const auto& tracer = imperative::GetCurrentTracer();
  imperative::NameVarBaseMap ins = {{"X", {var}}};
  framework::AttributeMap attrs = {{"in_dtype", var->DataType()},
                                   {"out_dtype", dst_type}};
  auto out = std::shared_ptr<imperative::VarBase>(
      new imperative::VarBase(tracer->GenerateUniqueName()));
  imperative::NameVarBaseMap outs = {{"Out", {out}}};

  {
    AutoCastGuard guard(tracer, false);
    tracer->TraceOp("cast", ins, outs, std::move(attrs));
  }

  return out;
}

static inline std::shared_ptr<imperative::VarBase> CastToFP16(
    const std::shared_ptr<VarBase>& var) {
  auto dst_type = framework::proto::VarType::FP16;
  if (NeedCast(var) && (var->DataType() != dst_type)) {
    return CastToType(var, dst_type);
  }
  return var;
}

static inline std::shared_ptr<imperative::VarBase> CastToFP32(
    const std::shared_ptr<VarBase>& var) {
  auto dst_type = framework::proto::VarType::FP32;
  if (NeedCast(var) && (var->DataType() != dst_type)) {
    return CastToType(var, dst_type);
  }
  return var;
}

static inline framework::proto::VarType::Type GetPromoteType(
    const NameVarBaseMap& ins) {
  auto dst_type = framework::proto::VarType::FP16;
  for (const auto& pair : ins) {
    for (const auto& var : pair.second) {
      if (var->DataType() == framework::proto::VarType::FP32) {
        dst_type = var->DataType();
        break;
      }
    }
  }
  return dst_type;
}

NameVarBaseMap AutoCastInputs(const std::string& op_type,
                              const NameVarBaseMap& ins) {
  NameVarBaseMap new_ins = {};
  if (g_white_ops.count(op_type)) {
    for (const auto& pair : ins) {
      VLOG(5) << "Op(" << op_type << "): Cast " << pair.first << " from "
              << GetDtypeStr(*pair.second.cbegin()) << " to float16";
      for (const auto& var : pair.second) {
        auto new_var = CastToFP16(var);
        new_ins[pair.first].emplace_back(new_var);
      }
    }
    return new_ins;
  } else if (g_black_ops.count(op_type)) {
    for (const auto& pair : ins) {
      VLOG(5) << "Op(" << op_type << "): Cast " << pair.first << " from "
              << GetDtypeStr(*pair.second.cbegin()) << " to float";
      for (const auto& var : pair.second) {
        auto new_var = CastToFP32(var);
        new_ins[pair.first].emplace_back(new_var);
      }
    }
    return new_ins;
  } else {
    auto dst_type = GetPromoteType(ins);

    for (const auto& pair : ins) {
      VLOG(5) << "Op(" << op_type << "): Cast " << pair.first << " from "
              << GetDtypeStr(*pair.second.cbegin()) << " to "
              << framework::DataTypeToString(dst_type);
      for (const auto& var : pair.second) {
        // NOTE(zhiqiu): Conv + BN always occur together, we needn't
        // cast X of batch_norm to FP32, which is produced by conv as FP16 type.
        if (op_type == "batch_norm" && pair.first == "X" &&
            dst_type == framework::proto::VarType::FP32) {
          new_ins[pair.first].emplace_back(var);
          continue;
        }
        auto new_var = dst_type == framework::proto::VarType::FP32
                           ? CastToFP32(var)
                           : CastToFP16(var);
        new_ins[pair.first].emplace_back(new_var);
      }
    }
    return new_ins;
  }
  return ins;
}

void SetAmpOpList(const std::unordered_set<std::string>& white_list,
                  const std::unordered_set<std::string>& black_list) {
  g_white_ops = white_list;
  g_black_ops = black_list;
}

std::tuple<const std::unordered_set<std::string>,
           const std::unordered_set<std::string>>
GetAmpOpList() {
  return std::make_tuple(g_white_ops, g_black_ops);
}

}  // namespace imperative
}  // namespace paddle
