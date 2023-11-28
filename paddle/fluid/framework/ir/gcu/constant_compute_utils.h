// Copyright (c) 2023 Enflame Authors. All Rights Reserved.
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
#include <string>
#include <unordered_map>
#include <vector>
#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/ir/node.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/scope.h"

namespace paddle {
namespace framework {
namespace ir {
using LoDTensor = phi::DenseTensor;
using ScopePtr = std::shared_ptr<paddle::framework::Scope>;
using PredAndPrepareFunc =
    std::function<bool(const ScopePtr &scope, const Node *node)>;

static bool IsUnknownShape(const std::vector<int64_t> &shape) {
  return std::any_of(
      shape.begin(), shape.end(), [](int64_t i) { return i < 0; });
}

static void SetShapeAttrIfNecessary(const Node *node) {
  auto op_desc = node->Op();
  if (op_desc->HasAttr("shape") && node->outputs.size() == 1) {
    std::vector<int64_t> shape;
    auto shape_attr = op_desc->GetAttr("shape");
    if (phi::enforce::demangle(shape_attr.type().name()) ==
        "std::vector<int, std::allocator<int> >") {
      auto origin_shape =
          PADDLE_GET_CONST(std::vector<int>, op_desc->GetAttr("shape"));
      std::for_each(origin_shape.begin(), origin_shape.end(), [&](int dim) {
        shape.emplace_back(static_cast<int64_t>(dim));
      });
    } else {
      shape = PADDLE_GET_CONST(std::vector<int64_t>, op_desc->GetAttr("shape"));
    }
    if (IsUnknownShape(shape)) {
      if (phi::enforce::demangle(shape_attr.type().name()) ==
          "std::vector<int, std::allocator<int> >") {
        auto var_shape = node->outputs[0]->Var()->GetShape();
        std::vector<int32_t> shape(var_shape.begin(), var_shape.end());
        op_desc->SetAttr("shape", shape);
      } else {
        op_desc->SetAttr("shape", node->outputs[0]->Var()->GetShape());
      }
      VLOG(3) << "Set attr shape for Op (" << node->Name() << ")";
    }
  }
}

static void DummyTensorForAllInputs(const ScopePtr &scope, const Node *node) {
  for (auto input : node->inputs) {
    if (input->IsCtrlVar()) {
      continue;
    }
    auto var = scope->GetVar(input->Name());
    auto tensor = var->GetMutable<LoDTensor>();
    tensor->mutable_data(
        platform::CPUPlace(),
        framework::TransToPhiDataType(input->Var()->GetDataType()));
    VLOG(3) << "Init tensor for var (" << input->Name() << ")";
  }
}

static bool AllInputsAreInitialized(const ScopePtr &scope, const Node *node) {
  for (auto input : node->inputs) {
    if (input->IsCtrlVar()) {
      continue;
    }
    auto var = scope->GetVar(input->Name());
    auto tensor = var->GetMutable<LoDTensor>();
    if (!(tensor->IsInitialized())) {
      return false;
    }
  }
  return true;
}

const auto kAlwaysTrueWithoutPrepare = [](const ScopePtr &scope,
                                          const Node *node) { return true; };

const auto kShapePredAndPrepareFunc = [](const ScopePtr &scope,
                                         const Node *node) {
  DummyTensorForAllInputs(scope, node);
  return true;
};

const auto kSlicePredWithoutPrepareFunc = [](const ScopePtr &scope,
                                             const Node *node) {
  return AllInputsAreInitialized(scope, node);
};

const std::unordered_map<std::string, PredAndPrepareFunc> kConstComputeOps = {
    {"fill_constant", kAlwaysTrueWithoutPrepare},
    {"shape", kShapePredAndPrepareFunc},
    {"slice", kSlicePredWithoutPrepareFunc}};

static bool NeedConstComputing(const ScopePtr &scope, const Node *node) {
  auto iter = kConstComputeOps.find(node->Op()->Type());
  if (iter == kConstComputeOps.end()) {
    return false;
  }
  return iter->second(scope, node);
}

class ConstantComputation {
 public:
  static void ConstComputeIfNecessary(const std::unique_ptr<OperatorBase> &op,
                                      const ScopePtr &scope,
                                      const Node *node) {
    SetShapeAttrIfNecessary(node);
    if (NeedConstComputing(scope, node)) {
      VLOG(3) << "Start constant computing for Op (" << node->Name() << ")";
      op->Run(*scope, paddle::platform::CPUPlace());
    }
  }
};
}  // namespace ir
}  // namespace framework
}  // namespace paddle
