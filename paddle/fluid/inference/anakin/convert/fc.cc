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
#include "paddle/fluid/inference/anakin/convert/helper.h"
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
  auto weight_tensor = tensor_from_var(*y_v, platform::CPUPlace());
  auto weight_shape = framework::vectorize2int(weight_tensor->dims());

  int out_dim = weight_shape[1];
  const int w_m = weight_shape[0];
  const int w_k = weight_shape[1];

  auto input_name = op_desc.Input(i_name).front();
  auto output_name = op_desc.Output("Out").front();

  this->engine_->AddOp(op_name, "Dense", {input_name}, {output_name});
  this->engine_->AddOpAttr(op_name, "bias_term", with_bias);
  this->engine_->AddOpAttr(op_name, "axis", 1);
  this->engine_->AddOpAttr(op_name, "out_dim", out_dim);

  auto *weight_data = weight_tensor->data<float>();
  PADDLE_ENFORCE(w_m * w_k == weight_tensor->numel());

  std::vector<float> trans_weight_data(weight_tensor->numel());
  for (int i = 0; i < w_m; i++) {
    for (int j = 0; j < w_k; j++) {
      trans_weight_data[i + j * w_m] = weight_data[i * w_k + j];
    }
  }

  auto *weight1 = pblock_from_vector<TargetT>(trans_weight_data);
  this->engine_->AddOpAttr(op_name, "weight_1", *weight1);

  // get bias
  if (with_bias) {
    auto *b_v = scope.FindVar(op_desc.Input("Bias").front());
    PADDLE_ENFORCE_NOT_NULL(b_v);
    auto weight2 = pblock_from_var<TargetT>(*b_v);
    this->engine_->AddOpAttr(op_name, "weight_2", *weight2);
  }
}

}  // namespace anakin
}  // namespace inference
}  // namespace paddle

#ifdef PADDLE_WITH_CUDA
REGISTER_CUDA_ANAKIN_OP_CONVERTER(mul, MulOpConverter<::anakin::saber::NV>);
REGISTER_CUDA_ANAKIN_OP_CONVERTER(fc, FcOpConverter<::anakin::saber::NV>);
#endif

REGISTER_CPU_ANAKIN_OP_CONVERTER(mul, MulOpConverter<::anakin::saber::X86>);
REGISTER_CPU_ANAKIN_OP_CONVERTER(fc, FcOpConverter<::anakin::saber::X86>);
