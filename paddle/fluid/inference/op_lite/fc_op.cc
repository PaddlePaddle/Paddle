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

#include "paddle/fluid/inference/op_lite/fc_op.h"
#include <vector>
#include "paddle/fluid/operators/math/fc_compute.h"

namespace paddle {
namespace inference {
namespace op_lite {
using framework::LoDTensor;
using framework::Variable;

bool FC::CheckShape() const {
  CHECK_OR_FALSE(param_.input);
  CHECK_OR_FALSE(param_.output);
  CHECK_OR_FALSE(param_.w);
  // bias is optional.

  const auto input_dims = param_.input->dims();
  const auto w_dims = param_.w->dims();

  if (param_.bias) {
    const auto bias_dims = param_.bias->dims();
    if (bias_dims.size() == 2) {
      CHECK_OR_FALSE(bias_dims[0] == 1);
      CHECK_OR_FALSE(bias_dims[1] == w_dims[1]);
    } else if (bias_dims.size() == 1) {
      CHECK_OR_FALSE(bias_dims[0] == w_dims[1]);
    }
  }

  CHECK_OR_FALSE(w_dims.size() == 2UL);
  CHECK_OR_FALSE(input_dims.size() > param_.in_num_col_dims);

  auto in_mat_dims =
      framework::flatten_to_2d(input_dims, param_.in_num_col_dims);
  CHECK_OR_FALSE(in_mat_dims[1] == w_dims[0]);

  return true;
}

bool FC::InferShape() const {
  const auto input_dims = param_.input->dims();
  const auto w_dims = param_.w->dims();

  // Set output dims
  std::vector<int64_t> output_dims(param_.in_num_col_dims + 1, 0UL);
  for (int i = 0; i < param_.in_num_col_dims; ++i) {
    output_dims[i] = input_dims[i];
  }
  output_dims.back() = w_dims[1];

  // share LoD
  param_.output->set_lod(param_.input->lod());
  return true;
}

bool FC::Run() {
  auto output_dims = param_.output->dims();
  auto w_dims = param_.w->dims();
  using T = float;

  int M = framework::product(output_dims) / output_dims[output_dims.size() - 1];
  auto blas = operators::math::BlasT<platform::CPUDeviceContext, T>(
      platform::CPUDeviceContext());

  operators::math::FCCompute<platform::CPUDeviceContext, float>(
      blas, M, w_dims[1], w_dims[0], param_.input->data<T>(),
      param_.w->data<T>(), param_.output->mutable_data<T>(platform::CPUPlace()),
      param_.bias ? param_.bias->data<T>() : nullptr);

  return false;
}

bool FC::Build(const framework::OpDesc &opdesc, framework::Scope *scope) {
  const auto &inputs = opdesc.Inputs();
  CHECK_OR_FALSE(inputs.count("Input"));
  CHECK_OR_FALSE(inputs.count("W"));
  CHECK_OR_FALSE(inputs.count("Bias"));
  CHECK_OR_FALSE(inputs.count("Out"));

  auto input = scope->FindVar(inputs.at("Input").front());
  auto w = scope->FindVar(inputs.at("W").front());
  auto output = scope->FindVar(inputs.at("Output").front());

  param_.input = input->GetMutable<LoDTensor>();
  param_.output = output->GetMutable<LoDTensor>();
  param_.w = w->GetMutable<LoDTensor>();

  if (!inputs.at("Bias").empty()) {
    auto bias = scope->FindVar(inputs.at("Bias").front());
    param_.bias = bias->GetMutable<LoDTensor>();
  }
}

}  // namespace op_lite
}  // namespace inference
}  // namespace paddle
