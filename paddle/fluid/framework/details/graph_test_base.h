// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include <algorithm>
#include <iostream>
#include <iterator>
#include <string>
#include "glog/logging.h"
#include "gtest/gtest.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/program_desc.h"

namespace paddle {
namespace framework {

class DummyOp : public OperatorBase {
 public:
  DummyOp(const std::string& type, const VariableNameMap& inputs,
          const VariableNameMap& outputs, const AttributeMap& attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

 private:
  void RunImpl(const Scope& scope,
               const platform::Place& place) const override {}
};

class SumOpMaker : public OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("X", "").AsDuplicable();
    AddOutput("Out", "");
    AddComment("");
  }
};

class AssignOpMaker : public OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("X", "").AsDuplicable();
    AddOutput("Out", "");
    AddComment("");
  }
};

class SplitOpMaker : public OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("X", "");
    AddOutput("Out", "").AsDuplicable();
    AddComment("");
  }
};

class DummyVarTypeInference : public VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext* ctx) const override {
    auto& inputs = ctx->Input("X");
    auto type = ctx->GetType(inputs.front());
    auto out_var_name = ctx->Output("Out").front();
    ctx->SetType(out_var_name, type);
  }
};

}  // namespace framework
}  // namespace paddle
