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

void FcOpConverter::operator()(const framework::proto::OpDesc &op,
                               const framework::Scope &scope, bool test_mode) {
  framework::OpDesc op_desc(op, nullptr);
  PADDLE_ENFORCE_EQ(op_desc.Input("X").size(), 1);
  PADDLE_ENFORCE_EQ(op_desc.Input("Y").size(), 1);
  PADDLE_ENFORCE_EQ(op_desc.Output("Out").size(), 1);

  auto x_name = op_desc.Input("X").front();
  auto op_name = op_desc.Type() + ":" + op_desc.Output("Out").front();
  auto *y_v = scope.FindVar(op_desc.Input("Y").front());
  PADDLE_ENFORCE_NOT_NULL(y_v);
  auto *y_t = y_v->GetMutable<framework::LoDTensor>();

  auto input_name = op_desc.Input("X").front();
  auto output_name = op_desc.Output("Out").front();

  auto weight_shape = framework::vectorize2int(y_t->dims());
  engine_->AddOp(op_name, "Dense", {input_name}, {output_name});
  engine_->AddOpAttr(op_name, "bias_term", false);
  engine_->AddOpAttr(op_name, "axis", 1);
  int out_dim = weight_shape[1];
  engine_->AddOpAttr(op_name, "out_dim", out_dim);

  weight_shape.push_back(1);
  weight_shape.push_back(1);
  Shape anakin_shape(weight_shape);

  framework::LoDTensor weight_tensor;
  weight_tensor.Resize(y_t->dims());
  TensorCopySync((*y_t), platform::CPUPlace(), &weight_tensor);

  auto *weight1 =
      GraphGlobalMem<NV>::Global().template new_block<AK_FLOAT>(anakin_shape);
  float *cpu_data = static_cast<float *>(weight1->h_tensor().mutable_data());
  std::copy_n(weight_tensor.data<float>(), weight_tensor.numel(), cpu_data);
  weight1->d_tensor().set_shape(anakin_shape);
  weight1->d_tensor().copy_from(weight1->h_tensor());
  engine_->AddOpAttr(op_name, "weight_1", *weight1);
}

}  // namespace anakin
}  // namespace inference
}  // namespace paddle
