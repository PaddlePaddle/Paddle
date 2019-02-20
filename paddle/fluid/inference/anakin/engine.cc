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
#include "framework/core/net/net.h"
#include "framework/graph/graph.h"

using anakin::Precision;
using anakin::OpRunType;
template <typename T, Precision P, OpRunType O>
using AnakinNetT = anakin::Net<T, P, O>;

template <typename T, Precision P>
using AnakinGraphT = anakin::graph::Graph<T, P>;

namespace paddle {
namespace inference {
namespace anakin {

template <typename TargetT, Precision PrecisionType, OpRunType RunType>
AnakinEngine<TargetT, PrecisionType, RunType>::AnakinEngine()
    : graph_(new AnakinGraphT<TargetT, PrecisionType>()),
      engine_(new AnakinNetT<TargetT, PrecisionType, RunType>()) {
  engine_->init(*graph_);
}

template <typename TargetT, Precision PrecisionType, OpRunType RunType>
void AnakinEngine<TargetT, PrecisionType, RunType>::DeclareInputs(
    const std::vector<std::string> &inputs) {
  inputs_ = inputs;
  std::sort(inputs_.begin(), inputs_.end());
}

template <typename TargetT, Precision PrecisionType, OpRunType RunType>
void AnakinEngine<TargetT, PrecisionType, RunType>::DeclareOutputs(
    const std::vector<std::string> &outputs) {
  outputs_ = outputs;
  std::sort(outputs_.begin(), outputs_.end());
}

template <typename TargetT, Precision PrecisionType, OpRunType RunType>
void AnakinEngine<TargetT, PrecisionType, RunType>::AddOp(
    const std::string &name, const std::string &type,
    const std::vector<std::string> &inputs,
    const std::vector<std::string> &outputs) {
  PADDLE_ENFORCE(graph_->AddOp(name, type, inputs, outputs), "Add operation.");
}

template <typename TargetT, Precision PrecisionType, OpRunType RunType>
std::vector<Tensor> AnakinEngine<TargetT, PrecisionType, RunType>::Execute(
    const std::vector<Tensor *> &inputs) {
  PADDLE_ENFORCE(inputs.empty() == false);
  for (auto input : inputs) {
    auto name = input->name();
    auto anakin_input = engine_->get_in(name);
    PADDLE_ENFORCE(anakin_input->shape().size() == input->shape().size());
    auto input_size =
        std::accumulate(input->shape().begin(), input->shape().end(), 1,
                        std::multiplies<int>());
    if (input_size > anakin_input->shape().count()) {
      graph_->Reshape(input->name(), input->shape());
      engine_.reset(new NetT(*graph_, true));
      anakin_input = engine_->get_in(name);
    }

    ::anakin::saber::Shape tmp_shape;
    std::copy(input->shape().begin(), input->shape().end(), tmp_shape.begin());
    anakin_input->reshape(tmp_shape);
    Place input_place;
    float *input_data = input->data<float>(&input_place, &input_size);
#ifdef PADDLE_WITH_CUDA
    if (std::is_same<TargetT, ::anakin::saber::NV>::value) {
      if (input_place == Place::kCpu) {
        cudaMemcpy(anakin_input->mutable_data(), input_data, input_size,
                   cudaMemcpyHostToDevice);
      } else {
        // anakin_input->mutable_data();
      }
    }
#endif
    if (std::is_same<TargetT, ::anakin::saber::X86>::value) {
      std::memcpy(anakin_input->mutable_data(), input_data, input_size);
    }
  }

#ifdef PADDLE_WITH_CUDA
  cudaDeviceSynchronize();
#endif
  engine_->prediction();
#ifdef PADDLE_WITH_CUDA
  cudaDeviceSynchronize();
#endif

  std::vector<Tensor> outputs;
  for (auto name : outputs_) {
    auto *anakin_out = engine_->get_out(name);
    std::vector<int> shape;
    auto valid_shape = anakin_out->valid_shape();
    std::copy(valid_shape.begin(), valid_shape.end(), shape.begin());
    Tensor output;
    output.Reshape(shape);
#ifdef PADDLE_WITH_CUDA
    if (std::is_same<TargetT, ::anakin::saber::NV>::value) {
      auto *data = output.mutable_data<float>(Place::kGpu);
      PADDLE_ENFORCE(cudaMemcpy(data, anakin_out->mutable_data(),
                                anakin_out->valid_size() * sizeof(float),
                                cudaMemcpyDeviceToHost));
    }
#endif
    if (std::is_same<TargetT, ::anakin::saber::X86>::value) {
      auto *data = output.mutable_data<float>(Place::kCpu);
      std::memcpy(data, anakin_out->mutable_data(),
                  anakin_out->valid_size() * sizeof(float));
    }
    outputs.push_back(output);
  }

  return outputs;
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
  auto *engine = new AnakinEngine();
  engine->engine_ = std::move(engine_->Clone());
  return std::unique_ptr<AnakinEngine>(engine);
}

void Tensor::Reshape(const std::vector<int> &shape) { shape_ = shape; }

const std::vector<int> &Tensor::shape() const { return shape_; }

void Tensor::SetName(const std::string &name) { name_ = name; }

const std::string &Tensor::name() const { return name_; }

void Tensor::SetDataType(const DataType dtype) { dtype_ = dtype; }

DataType Tensor::dtype() const { return dtype_; }

}  // namespace anakin
}  // namespace inference
}  // namespace paddle
