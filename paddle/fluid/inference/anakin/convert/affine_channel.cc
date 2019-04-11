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

#include "paddle/fluid/inference/anakin/convert/affine_channel.h"
#include <algorithm>
#include <string>
#include <vector>

using anakin::graph::GraphGlobalMem;
using anakin::AK_FLOAT;
using anakin::Precision;
using anakin::saber::NV;
using anakin::saber::X86;
using anakin::saber::Shape;
using anakin::PBlock;
using anakin::PTuple;

namespace paddle {
namespace inference {
namespace anakin {

void AffineChannelOpConverter::operator()(
    const framework::proto::OpDesc &op, const framework::BlockDesc &block_desc,
    const framework::Scope &scope, bool test_mode) {
  framework::OpDesc op_desc(op, nullptr);
  PADDLE_ENFORCE_EQ(op_desc.Input("X").size(), 1);
  PADDLE_ENFORCE_EQ(op_desc.Output("Out").size(), 1);

  auto op_name = op_desc.Type() + ":" + op_desc.Output("Out").front();

  auto input_name = op_desc.Input("X").front();
  auto output_name = op_desc.Output("Out").front();

  // Copy the Scale to CPUPlace and get the pointer.
  auto *scale_v = scope.FindVar(op_desc.Input("Scale").front());
  PADDLE_ENFORCE_NOT_NULL(scale_v);
  auto *scale_t = scale_v->GetMutable<framework::LoDTensor>();
  std::unique_ptr<framework::LoDTensor> scale_tensor(
      new framework::LoDTensor());
  scale_tensor->Resize(scale_t->dims());
  TensorCopySync((*scale_t), platform::CPUPlace(), scale_tensor.get());

  // Copy the Bias to CPUPlace and get the pointer.
  auto *bias_v = scope.FindVar(op_desc.Input("Bias").front());
  PADDLE_ENFORCE_NOT_NULL(bias_v);
  auto *bias_t = bias_v->GetMutable<framework::LoDTensor>();
  std::unique_ptr<framework::LoDTensor> bias_tensor(new framework::LoDTensor());
  bias_tensor->Resize(bias_t->dims());
  TensorCopySync((*bias_t), platform::CPUPlace(), bias_tensor.get());

  engine_->AddOp(op_name, "AffineChannel", {input_name}, {output_name});

  // Generate the Scale parameter of Anakin.
  auto scale_shape = framework::vectorize2int(scale_t->dims());
  while (scale_shape.size() < 4) {
    scale_shape.insert(scale_shape.begin(), 1);
  }
  Shape anakin_scale_shape(scale_shape);
  auto *weight1 = GraphGlobalMem<NV>::Global().template new_block<AK_FLOAT>(
      anakin_scale_shape);
  float *scale_cpu_data =
      static_cast<float *>(weight1->h_tensor().mutable_data());
  std::copy_n(scale_tensor->data<float>(), scale_tensor->numel(),
              scale_cpu_data);
  weight1->d_tensor().set_shape(anakin_scale_shape);
  weight1->d_tensor().copy_from(weight1->h_tensor());
  engine_->AddOpAttr(op_name, "weight_1", *weight1);

  // Generate the Bias parameter of Anakin.
  auto bias_shape = framework::vectorize2int(bias_t->dims());
  while (bias_shape.size() < 4) {
    bias_shape.insert(bias_shape.begin(), 1);
  }
  Shape anakin_bias_shape(bias_shape);
  auto *weight2 = GraphGlobalMem<NV>::Global().template new_block<AK_FLOAT>(
      anakin_bias_shape);
  float *bias_cpu_data =
      static_cast<float *>(weight2->h_tensor().mutable_data());
  std::copy_n(bias_tensor->data<float>(), bias_tensor->numel(), bias_cpu_data);
  weight2->d_tensor().set_shape(anakin_bias_shape);
  weight2->d_tensor().copy_from(weight2->h_tensor());
  engine_->AddOpAttr(op_name, "weight_2", *weight2);
}

}  // namespace anakin
}  // namespace inference
}  // namespace paddle

REGISTER_ANAKIN_OP_CONVERTER(affine_channel, AffineChannelOpConverter);
