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
AnakinEngine<TargetT, PrecisionType, RunType>::AnakinEngine(
    bool need_summary, int device, int max_batch_size,
    std::map<std::string, std::vector<int>> max_input_shape)
    : graph_(new AnakinGraphT<TargetT, PrecisionType>()),
      net_(new AnakinNetT<TargetT, PrecisionType, RunType>(need_summary)) {
  device_ = device;
  max_batch_size_ = max_batch_size;
  max_input_shape_ = max_input_shape;
}

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
    const std::map<std::string, framework::LoDTensor *> &outputs,
    cudaStream_t stream) {
  cudaDeviceSynchronize();
  for (const auto &input : inputs) {
    auto *tensor = input.second;
    auto *data = tensor->data<float>();

    auto fluid_input_shape = framework::vectorize2int(tensor->dims());
    while (fluid_input_shape.size() < 4) {
      fluid_input_shape.push_back(1);
    }
    auto *anakin_input = net_->get_in(input.first);
    std::vector<int> max_input_shape = max_input_shape_[input.first];
    int max_shape_sum =
        std::accumulate(max_input_shape.begin(), max_input_shape.end(), 1,
                        std::multiplies<int>());

    PADDLE_ENFORCE(max_shape_sum >= tensor->numel(),
                   "The anakin input max shape should be greater than"
                   " or equal to the real input shape, Please set the max "
                   "input shape using EnableAnakinEngine");
    anakin_input->reshape(fluid_input_shape);
    ::anakin::saber::Tensor<TargetT> tmp_anakin_tensor(data, TargetT(), 0,
                                                       fluid_input_shape);
    anakin_input->copy_from(tmp_anakin_tensor);
  }
  net_->prediction();
  cudaDeviceSynchronize();
  for (const auto &output : outputs) {
    platform::CUDAPlace gpu_place(device_);
    auto *tensor = output.second;
    auto *anakin_output = net_->get_out(output.first);
    auto *anakin_data = anakin_output->data();
    auto anakin_output_shape = anakin_output->valid_shape();
    tensor->Resize(framework::make_ddim(anakin_output_shape));
    auto *fluid_data = tensor->mutable_data<float>(gpu_place);
    memory::Copy(gpu_place, static_cast<void *>(fluid_data), gpu_place,
                 static_cast<void *>(anakin_data),
                 tensor->numel() * sizeof(float), stream);
  }
  cudaDeviceSynchronize();
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
