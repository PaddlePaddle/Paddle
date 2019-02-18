// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/inference/anakin/engine.h"

namespace paddle {
namespace inference {
namespace anakin {
using anakin::Precision;
using anakin::OpRunType;

template <typename TargetT, Precision PrecisionType, OpRunType RunType>
AnakinEngine<TargetT, PrecisionType, RunType>::AnakinEngine()
    : graph_(new Graph<TargetT, PrecisionType>()),
      engine_(new Net<TargetT, PrecisionType, RunType>()) {
  engine_->init(*graph_);
}

template <typename TargetT, Precision PrecisionType, OpRunType RunType>
void AnakinEngine<TargetT, PrecisionType, RunType>::DeclareInputs(
    const std::vector<std::string>> &inputs) {
  inputs_ = std::sort(inputs.begin(), inputs.end());
}

template <typename TargetT, Precision PrecisionType, OpRunType RunType>
void AnakinEngine<TargetT, PrecisionType, RunType>::DeclareOutputs(
    const std::vector<std::string> &outputs) {
  outputs_ = std::sort(outputs_.begin(), outputs_.end());
}

template <typename TargetT, Precision PrecisionType, OpRunType RunType>
void AnakinEngine<TargetT, PrecisionType, RunType>::Execute(
    const std::vector<Tensor *> &inputs, std::vector<Tensor *> *outputs) {
  for (auto input : inputs) {
    //
  }
  auto anakin_inputs = engine_->get_in_list();

  auto anakin_outputs = engine_->get_out_list();
  engine_->prediction();
}

template <typename TargetT, Precision PrecisionType, OpRunType RunType>
void AnakinEngine<TargetT, PrecisionType, RunType>::AddOp(
    const std::string &name, const std::string &type,
    const std::vector<std::string> &inputs,
    const std::vector<std::string> &outputs) {
  PADDLE_ENFORCE(graph_->AddOp(name, type, inputs, outputs), "Add operation.");
}

template <typename TargetT, Precision PrecisionType, OpRunType RunType>
void AnakinEngine<TargetT, PrecisionType, RunType>::AddVar(
    const std::string &id, DataType dtype, const std::vector<int> &shape) {
  //
}

template <typename TargetT, Precision PrecisionType, OpRunType RunType>
void AnakinEngine<TargetT, PrecisionType, RunType>::FreezeNetwork() {
  PADDLE_ENFORCE(graph_->Freeze(), "Freeze anakin subgraph.");
  PADDLE_ENFORCE(graph_->Optimize(), "Graph optimization.");

  std::vector<std::string> inputs = graph_->get_ins();
  std::sort(inputs.begin(), inputs.end());
  PADDLE_ENFORCE(inputs_ == inputs);

  std::vector<std::string> outputs = graph_->get_outs();
  std::sort(outputs.begin(), outputs.end());
  PADDLE_ENFORCE(outputs_ == outputs);
}

template <typename TargetT, Precision PrecisionType, OpRunType RunType>
std::unique_ptr<AnakinEngine<TargetT, PrecisionType, RunType>>
AnakinEngine<TargetT, PrecisionType, RunType>::Clone() {
  return new AnakinEngine(*this);
}

template <typename TargetT, Precision PrecisionType, OpRunType RunType>
AnakinEngine<TargetT, PrecisionType, RunType>::AnakinEngine(
    const AnakinEngine<TargetT, PrecisionType, RunType> &engine) {
  engine_ = std::move(engine->engine_->Clone());
}

void Tensor::Resize(const std::vector<int> &shape) { shape_ = shape; }

void Tensor::SetName(const std::string &name) { name_ = name; }

const std::string &Tensor::name() const { return name_; }

DataType Tensor::dtype() const { return dtype_; }

const std::vector<int> &shape() const { return shape_; }
}
}
}
