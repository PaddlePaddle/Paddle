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
bool AnakinEngine<TargetT, PrecisionType, RunType>::IsOpSupported(
    const std::string &op_type, const attrs_t &attrs) {
  return true;
}

template <typename TargetT, Precision PrecisionType, OpRunType RunType>
void AnakinEngine<TargetT, PrecisionType, RunType>::DeclareInput(
    const std::string &name, const Tensor *tensor, const attrs_t &attrs) {
  inputs_.insert({name, tensor});
}

template <typename TargetT, Precision PrecisionType, OpRunType RunType>
void AnakinEngine<TargetT, PrecisionType, RunType>::DeclareOutput(
    const std::string &name, const Tensor *tensor, const attrs_t &attrs) {}

template <typename TargetT, Precision PrecisionType, OpRunType RunType>
void AnakinEngine<TargetT, PrecisionType, RunType>::Execute(int batch_size) {
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
    const std::string &id, DataType dtype, const shape_t &shape) {
  //
}

template <typename TargetT, Precision PrecisionType, OpRunType RunType>
void AnakinEngine<TargetT, PrecisionType, RunType>::FreezeNetwork() {
  PADDLE_ENFORCE(graph_->Freeze(), "Freeze anakin subgraph.");
  PADDLE_ENFORCE(graph_->Optimize(), "Graph optimization.");
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
}
}
}
