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

const std::unordered_set<std::string> fp16_ops = {
    "conv2d", "matmul", "mul",
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

  tracer->TraceOp("cast", ins, outs, std::move(attrs));
  return out;
}

std::shared_ptr<imperative::VarBase> CastToFP16(
    const std::shared_ptr<VarBase>& var) {
  return CastToType(var, framework::proto::VarType::FP16);
}

std::shared_ptr<imperative::VarBase> CastToFP32(
    const std::shared_ptr<VarBase>& var) {
  return CastToType(var, framework::proto::VarType::FP32);
}

NameVarBaseMap AutoCastInputs(const std::string& op_type,
                              const NameVarBaseMap& ins) {
  NameVarBaseMap new_ins = {};
  if (fp16_ops.count(op_type)) {
    for (const auto& pair : ins) {
      auto new_var = CastToFP16(*pair.second.cbegin());
      new_ins[pair.first].emplace_back(new_var);
    }
  }
  return new_ins;
}
}  // namespace imperative
}  // namespace paddle
