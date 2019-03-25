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
#include <algorithm>
#include <cstring>
#include <map>
#include <utility>
#include "paddle/fluid/framework/ddim.h"

using anakin::Precision;
using anakin::OpRunType;
using paddle::framework::LoDTensor;
template <typename T, Precision P, OpRunType O>
using AnakinNetT = anakin::Net<T, P, O>;

template <typename T, Precision P>
using AnakinGraphT = anakin::graph::Graph<T, P>;

namespace paddle {
namespace inference {
namespace anakin {

template <typename TargetT, Precision PrecisionType, OpRunType RunType>
AnakinEngine<TargetT, PrecisionType, RunType>::AnakinEngine(bool need_summary)
    : graph_(new AnakinGraphT<TargetT, PrecisionType>()),
      net_(new AnakinNetT<TargetT, PrecisionType, RunType>(need_summary)) {}

template <typename TargetT, Precision PrecisionType, OpRunType RunType>
AnakinEngine<TargetT, PrecisionType, RunType>::~AnakinEngine() {}

template <typename TargetT, Precision PrecisionType, OpRunType RunType>
void AnakinEngine<TargetT, PrecisionType, RunType>::SetInputShape(
    const std::string &name, std::vector<int> shape) {
  graph_->AddOpAttr<::anakin::PTuple<int>>(name, "input_shape",
                                           std::move(shape));
}

template <typename TargetT, Precision PrecisionType, OpRunType RunType>
void AnakinEngine<TargetT, PrecisionType, RunType>::InitGraph() {
  net_->init(*graph_);
}

template <typename TargetT, Precision PrecisionType, OpRunType RunType>
void AnakinEngine<TargetT, PrecisionType, RunType>::AddOp(
    const std::string &name, const std::string &type,
    const std::vector<std::string> &inputs,
    const std::vector<std::string> &outputs) {
  PADDLE_ENFORCE(graph_->AddOp(name, type, inputs, outputs), "Add operation.");
}

template <typename TargetT, Precision PrecisionType, OpRunType RunType>
void AnakinEngine<TargetT, PrecisionType, RunType>::Execute(
    const std::map<std::string, framework::LoDTensor *> &inputs,
    const std::map<std::string, framework::LoDTensor *> &outputs) {
  for (const auto &input : inputs) {
    auto *tensor = input.second;
    auto *data = tensor->data<float>();
    auto shape = framework::vectorize2int(tensor->dims());
    ::anakin::saber::Shape anakin_shape(shape);
    auto *anakin_input = net_->get_in(input.first);
    ::anakin::saber::Tensor<TargetT> tmp_anakin_tensor(data, TargetT(), 0,
                                                       anakin_shape);
    anakin_input->share_from(tmp_anakin_tensor);
  }

  for (const auto &output : outputs) {
    auto *tensor = output.second;
    auto *data = tensor->data<float>();
    auto shape = framework::vectorize2int(tensor->dims());
    ::anakin::saber::Shape anakin_shape(shape);
    auto *anakin_output = net_->get_out(output.first);
    ::anakin::saber::Tensor<TargetT> tmp_anakin_tensor(data, TargetT(), 0,
                                                       anakin_shape);
    anakin_output->share_from(tmp_anakin_tensor);
  }
  net_->prediction();
}

template <typename TargetT, Precision PrecisionType, OpRunType RunType>
void AnakinEngine<TargetT, PrecisionType, RunType>::Freeze() {
  PADDLE_ENFORCE(graph_->Freeze(), "Freeze anakin subgraph.");
}

template <typename TargetT, Precision PrecisionType, OpRunType RunType>
void AnakinEngine<TargetT, PrecisionType, RunType>::Optimize() {
  PADDLE_ENFORCE(graph_->Optimize(), "Graph optimization.");
}

template <typename TargetT, Precision PrecisionType, OpRunType RunType>
std::unique_ptr<AnakinEngine<TargetT, PrecisionType, RunType>>
AnakinEngine<TargetT, PrecisionType, RunType>::Clone() {
  auto *engine = new AnakinEngine();
  engine->net_ = std::move(net_->Clone());
  return std::unique_ptr<AnakinEngine>(engine);
}

template class AnakinEngine<::anakin::saber::NV, ::anakin::Precision::FP32>;
}  // namespace anakin
}  // namespace inference
}  // namespace paddle
