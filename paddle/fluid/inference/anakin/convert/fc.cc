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

#include "paddle/fluid/inference/anakin/convert/fc.h"
#include <algorithm>
#include <string>
#include <vector>

using anakin::graph::GraphGlobalMem;
using anakin::AK_FLOAT;
using anakin::saber::Shape;

namespace paddle {
namespace inference {
namespace anakin {

template <typename TargetT>
void FcBaseOpConverter<TargetT>::operator()(
    const framework::proto::OpDesc &op, const framework::BlockDesc &block_desc,
    const framework::Scope &scope, bool test_mode) {
  framework::OpDesc op_desc(op, nullptr);
  auto input_names = op_desc.InputNames();
  bool with_bias = input_names.size() == 3;

  std::string w_name = "Y";
  std::string i_name = "X";
  if (with_bias) {
    w_name = "W";
    i_name = "Input";
  }

  auto op_name = op_desc.Type() + ":" + op_desc.Output("Out").front();

  // get weights
  auto *y_v = scope.FindVar(op_desc.Input(w_name).front());
  PADDLE_ENFORCE_NOT_NULL(y_v);
  auto *y_t = y_v->GetMutable<framework::LoDTensor>();

  auto input_name = op_desc.Input(i_name).front();
  auto output_name = op_desc.Output("Out").front();

  this->engine_->AddOp(op_name, "Dense", {input_name}, {output_name});
  this->engine_->AddOpAttr(op_name, "bias_term", with_bias);
  this->engine_->AddOpAttr(op_name, "axis", 1);

  auto weight_shape = framework::vectorize2int(y_t->dims());
  int out_dim = weight_shape[1];
  this->engine_->AddOpAttr(op_name, "out_dim", out_dim);
  const int w_m = weight_shape[0];
  const int w_k = weight_shape[1];

  if (weight_shape.size() < 4UL) {
    weight_shape.insert(weight_shape.begin(), 4UL - weight_shape.size(), 1);
  }
  Shape anakin_shape(weight_shape);

  framework::LoDTensor weight_tensor;
  weight_tensor.Resize(y_t->dims());
  TensorCopySync((*y_t), platform::CPUPlace(), &weight_tensor);
  auto *weight_data = weight_tensor.data<float>();
  PADDLE_ENFORCE(w_m * w_k == weight_tensor.numel());

  std::vector<float> trans_weight_data(weight_tensor.numel());
  for (int i = 0; i < w_m; i++) {
    for (int j = 0; j < w_k; j++) {
      trans_weight_data[i + j * w_m] = weight_data[i * w_k + j];
    }
  }
  auto *weight1 =
      GraphGlobalMem<TargetT>::Global().template new_block<AK_FLOAT>(
          anakin_shape);
  float *cpu_data = static_cast<float *>(weight1->h_tensor().mutable_data());
  std::copy_n(trans_weight_data.data(), weight_tensor.numel(), cpu_data);
  weight1->d_tensor().set_shape(anakin_shape);
  weight1->d_tensor().copy_from(weight1->h_tensor());
  this->engine_->AddOpAttr(op_name, "weight_1", *weight1);

  // get bias
  if (with_bias) {
    auto *b_v = scope.FindVar(op_desc.Input("Bias").front());
    PADDLE_ENFORCE_NOT_NULL(b_v);
    auto *b_t = b_v->GetMutable<framework::LoDTensor>();

    auto bias_shape = framework::vectorize2int(b_t->dims());
    framework::LoDTensor bias_tensor;
    bias_tensor.Resize(b_t->dims());
    TensorCopySync((*b_t), platform::CPUPlace(), &bias_tensor);
    auto *bias_data = bias_tensor.data<float>();
    bias_shape.insert(bias_shape.begin(), 1);
    bias_shape.insert(bias_shape.begin(), 1);
    bias_shape.insert(bias_shape.begin(), 1);
    // bias_shape.push_back(1);
    // bias_shape.push_back(1);
    Shape anakin_bias_shape(bias_shape);

    auto *weight2 =
        GraphGlobalMem<TargetT>::Global().template new_block<AK_FLOAT>(
            anakin_bias_shape);
    float *cpu_data2 = static_cast<float *>(weight2->h_tensor().mutable_data());
    std::copy_n(bias_data, bias_tensor.numel(), cpu_data2);
    weight2->d_tensor().set_shape(anakin_bias_shape);
    weight2->d_tensor().copy_from(weight2->h_tensor());
    this->engine_->AddOpAttr(op_name, "weight_2", *weight2);
  }
}

}  // namespace anakin
}  // namespace inference
}  // namespace paddle

REGISTER_CUDA_ANAKIN_OP_CONVERTER(mul, MulOpConverter<::anakin::saber::NV>);
REGISTER_CUDA_ANAKIN_OP_CONVERTER(fc, FcOpConverter<::anakin::saber::NV>);
