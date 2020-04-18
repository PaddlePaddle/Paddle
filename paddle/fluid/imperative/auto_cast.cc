/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/imperative/auto_cast.h"
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include "paddle/fluid/imperative/layer.h"
#include "paddle/fluid/imperative/tracer.h"
#include "paddle/fluid/imperative/variable_wrapper.h"

namespace paddle {
namespace imperative {

// TODO(zhiqiu): consider to cache casted params for better performance.

// Operators that supports fp16 calculation can be divided into 3 sets.

// The set of ops that support fp16 calculation and are considered numerically -
// safe and performance - critical.These ops are always converted to fp16.
const std::unordered_set<std::string> white_ops = {
    "conv2d", "matmul", "mul",
};

// The set of ops that support fp16 calculation and are considered numerically -
// dangerous and whose effects may also be observed in downstream ops.
const std::unordered_set<std::string> black_ops = {
    "exp",
    "square",
    "log",
    "mean",
    "sum",
    "cos_sim",
    "softmax",
    "softmax_with_cross_entropy",
    "sigmoid_cross_entropy_with_logits",
    "cross_entropy",
    "cross_entropy2",
};

// This set contains two types of ops. All ops supported fp16 calculation. One
// of two types is considered numerically - safe, but may be made unsafe by an
// upstream blacklist op. Another type do not have numerically - significant
// effects, like stack, flatten2.
const std::unordered_set<std::string> gray_ops = {
    "elementwise_add",
    "elementwise_sub",
    "elementwise_mul",
    "elementwise_div",
    "elementwise_max",
    "elementwise_min",
    "elementwise_pow",
    "elementwise_mod",
    "elementwise_floordiv",
    "batch_norm",
    "tanh",
    "sigmoid",
    "lookup_table",
    "top_k",
    "pool2d",
    "pool3d",
    "dropout",
    "relu",
    "relu6",
    "leaky_relu",
    "soft_relu",
    "flatten2",
    "stack",
    "unstack",
    "uniform_random_batch_size_like",
    "gaussian_random",
    "gaussian_random_batch_size_like",
    "slice",
    "rank",
    "scale",
    "transpose2",
    "reshape2",
    "gather",
    "fill_constant",
    "get_tensor_from_selected_rows",
    "sign",
    "cast",
};

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

// TODO(zhiqiu): can we just cast the tensor, instead of tracing a cast op.
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
    for (const auto& op : pair.second) {
      if (op->DataType() == framework::proto::VarType::FP32) {
        dst_type = op->DataType();
      }
    }
  }
  return dst_type;
}

NameVarBaseMap AutoCastInputs(const std::string& op_type,
                              const NameVarBaseMap& ins) {
  NameVarBaseMap new_ins = {};
  if (white_ops.count(op_type)) {
    for (const auto& pair : ins) {
      VLOG(5) << "Cast " << pair.first << " " << pair.second.size() << " "
              << (*pair.second.cbegin())->DataType() << " to FP16";
      for (const auto& var : pair.second) {
        auto new_var = CastToFP16(var);
        new_ins[pair.first].emplace_back(new_var);
      }
    }
    return new_ins;
  } else if (black_ops.count(op_type)) {
    for (const auto& pair : ins) {
      VLOG(5) << "Cast " << pair.first << " " << pair.second.size() << " "
              << (*pair.second.cbegin())->DataType() << " to FP16";
      for (const auto& var : pair.second) {
        auto new_var = CastToFP32(var);
        new_ins[pair.first].emplace_back(new_var);
      }
    }
    return new_ins;
  } else if (gray_ops.count(op_type)) {
    auto dst_type = GetPromoteType(ins);

    for (const auto& pair : ins) {
      VLOG(5) << "Cast " << pair.first << " " << pair.second.size() << " "
              << (*pair.second.cbegin())->DataType() << " to " << dst_type;
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
}  // namespace imperative
}  // namespace paddle
