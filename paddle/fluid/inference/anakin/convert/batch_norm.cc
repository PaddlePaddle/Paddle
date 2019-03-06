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

#include "paddle/fluid/inference/anakin/convert/batch_norm.h"
#include <math.h>
#include <algorithm>
#include <map>
#include <string>

using anakin::graph::GraphGlobalMem;
using anakin::AK_FLOAT;
using anakin::saber::NV;
using anakin::saber::Shape;

namespace paddle {
namespace inference {
namespace anakin {

void BatchNormOpConverter::operator()(const framework::proto::OpDesc &op,
                                      const framework::Scope &scope,
                                      bool test_mode) {
  framework::OpDesc op_desc(op, nullptr);
  PADDLE_ENFORCE_EQ(op_desc.Output("Y").size(), 1);
  std::map<std::string, std::string> inputs;
  for (auto k : {"X", "Scale", "Bias", "Mean", "Variance"}) {
    PADDLE_ENFORCE_EQ(op_desc.Input(k).size(), 1UL);
    auto v = op_desc.Input(k).front();
    inputs.insert({k, v});
  }

  auto output = op_desc.Output("Y").front();
  auto op_name = op_desc.Type() + ":" + op_desc.Output("Y").front();
  engine_->AddOp(op_name, "Scale", {inputs["X"]}, {output});
  engine_->AddOpAttr(op_name, "bias_term", true);
  engine_->AddOpAttr(op_name, "axis", 1);
  engine_->AddOpAttr(op_name, "num_axes", 1);

  bool is_test = boost::get<bool>(op_desc.GetAttr("is_test"));
  PADDLE_ENFORCE(is_test);
  //float momentum = boost::get<float>(op_desc.GetAttr("momentum"));
  float epsilon = boost::get<float>(op_desc.GetAttr("epsilon"));
  engine_->AddOpAttr(op_name, "epsilon", epsilon);

  auto get_lod_tensor = [this, &scope, &op_name](
      const std::string &var_name, framework::LoDTensor *tensor) {
      auto *v = scope.FindVar(var_name);
      PADDLE_ENFORCE_NOT_NULL(v);
      auto *t = v->GetMutable<framework::LoDTensor>();
      tensor->Resize(t->dims());
      TensorCopySync(*t, platform::CPUPlace(), tensor);
  };

  framework::LoDTensor bias_t;
  framework::LoDTensor mean_t;
  framework::LoDTensor scale_t;
  framework::LoDTensor variance_t;
  get_lod_tensor(inputs["Bias"], &bias_t);
  get_lod_tensor(inputs["Mean"], &mean_t);
  get_lod_tensor(inputs["Scale"], &scale_t);
  get_lod_tensor(inputs["Variance"], &variance_t);

  auto *bias = bias_t.mutable_data<float>(platform::CPUPlace());
  auto *mean = mean_t.mutable_data<float>(platform::CPUPlace());
  auto *scale = scale_t.mutable_data<float>(platform::CPUPlace());
  auto *variance = variance_t.mutable_data<float>(platform::CPUPlace());

  framework::LoDTensor combile_scale_t;
  framework::LoDTensor combile_bias_t;
  combile_scale_t.Resize(scale_t.dims());
  combile_bias_t.Resize(bias_t.dims());

  auto *combile_scale = combile_scale_t.mutable_data<float>(platform::CPUPlace());
  auto *combile_bias = combile_bias_t.mutable_data<float>(platform::CPUPlace());

  size_t elem_num = combile_scale_t.memory_size() / sizeof(float);
  for (size_t i = 0; i < elem_num; i++) {
    combile_scale[i] = scale[i] / sqrtf(variance[i] + epsilon);
    combile_bias[i] = bias[i] - mean[i] * combile_scale[i];
  }

  auto fill_shape = [](size_t n, std::vector<int> *shape) {
    shape->insert(shape->begin(), 1);
    if (shape->size() < n) {
      shape->insert(shape->end(), n - shape->size(), 1);
    }
  };
  auto scale_shape = framework::vectorize2int(combile_scale_t.dims());
  auto bias_shape = framework::vectorize2int(combile_bias_t.dims());
  fill_shape(4, &scale_shape);
  fill_shape(4, &bias_shape);
  Shape weight1_shape(scale_shape);
  Shape weight2_shape(bias_shape);
  auto *weight1 =
      GraphGlobalMem<NV>::Global().template new_block<AK_FLOAT>(weight1_shape);
  auto *scale_data = static_cast<float *>(weight1->h_tensor().mutable_data());
  std::copy_n(combile_scale_t.data<float>(), combile_scale_t.numel(), scale_data);
  engine_->AddOpAttr(op_name, "weight_1", *weight1);

  auto *weight2 =
      GraphGlobalMem<NV>::Global().template new_block<AK_FLOAT>(weight2_shape);
  auto *bias_data = static_cast<float *>(weight2->h_tensor().mutable_data());
  std::copy_n(combile_bias_t.data<float>(), combile_bias_t.numel(), bias_data);
  engine_->AddOpAttr(op_name, "weight_2", *weight2);
}

}  // namespace anakin
}  // namespace inference
}  // namespace paddle

REGISTER_ANAKIN_OP_CONVERTER(batch_norm, BatchNormOpConverter);
