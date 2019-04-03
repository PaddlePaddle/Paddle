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
#include <vector>

using anakin::graph::GraphGlobalMem;
using anakin::AK_FLOAT;
using anakin::saber::Shape;

namespace paddle {
namespace inference {
namespace anakin {

template <typename TargetT>
void BatchNormOpConverter<TargetT>::operator()(
    const framework::proto::OpDesc &op, const framework::BlockDesc &block_desc,
    const framework::Scope &scope, bool test_mode) {
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
  auto epsilon = boost::get<float>(op_desc.GetAttr("epsilon"));
  // auto momentum = boost::get<float>(op_desc.GetAttr("momentum"));

  auto bn_op_name = op_name + ":bn";
  auto bn_output = bn_op_name + "_output";
  this->engine_->AddOp(bn_op_name, "BatchNorm", {inputs["X"]}, {bn_output});
  this->engine_->AddOpAttr(bn_op_name, "epsilon", epsilon);
  this->engine_->AddOpAttr(bn_op_name, "momentum", static_cast<float>(1.0));

  auto scale_op_name = op_name + ":scale";
  auto get_lod_tensor = [this, &scope, &op_name](const std::string &var_name,
                                                 framework::LoDTensor *tensor) {
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

  auto fill_shape = [](size_t n, std::vector<int> shape) {
    shape.insert(shape.begin(), 1);
    if (shape.size() < n) {
      shape.insert(shape.end(), n - shape.size(), 1);
    }
    return shape;
  };
  Shape shape1(fill_shape(4, framework::vectorize2int(mean_t.dims())));
  Shape shape2(fill_shape(4, framework::vectorize2int(variance_t.dims())));
  auto *weight1 =
      GraphGlobalMem<TargetT>::Global().template new_block<AK_FLOAT>(shape1);
  auto *mean_data = static_cast<float *>(weight1->h_tensor().mutable_data());
  std::copy_n(mean_t.data<float>(), mean_t.numel(), mean_data);
  this->engine_->AddOpAttr(bn_op_name, "weight_1", *weight1);

  auto *weight2 =
      GraphGlobalMem<TargetT>::Global().template new_block<AK_FLOAT>(shape2);
  auto *variance_data =
      static_cast<float *>(weight2->h_tensor().mutable_data());
  std::copy_n(variance_t.data<float>(), variance_t.numel(), variance_data);
  this->engine_->AddOpAttr(bn_op_name, "weight_2", *weight2);

  Shape shape3(std::vector<int>({1, 1, 1, 1}));
  auto *weight3 =
      GraphGlobalMem<TargetT>::Global().template new_block<AK_FLOAT>(shape3);
  auto *alpha_data = static_cast<float *>(weight3->h_tensor().mutable_data());
  float weight3_data[] = {1};
  std::copy(std::begin(weight3_data), std::end(weight3_data), alpha_data);
  this->engine_->AddOpAttr(bn_op_name, "weight_3", *weight3);

  Shape scale_shape(fill_shape(4, framework::vectorize2int(scale_t.dims())));
  auto *scale = GraphGlobalMem<TargetT>::Global().template new_block<AK_FLOAT>(
      scale_shape);
  auto *scale_data = static_cast<float *>(scale->h_tensor().mutable_data());
  std::copy_n(scale_t.data<float>(), scale_t.numel(), scale_data);

  Shape bias_shape(fill_shape(4, framework::vectorize2int(bias_t.dims())));
  auto *bias = GraphGlobalMem<TargetT>::Global().template new_block<AK_FLOAT>(
      bias_shape);
  auto *bias_data = static_cast<float *>(bias->h_tensor().mutable_data());
  std::copy_n(bias_t.data<float>(), bias_t.numel(), bias_data);

  this->engine_->AddOp(scale_op_name, "Scale", {bn_output}, {output});
  this->engine_->AddOpAttr(scale_op_name, "axis", 1);
  this->engine_->AddOpAttr(scale_op_name, "num_axes", 1);
  this->engine_->AddOpAttr(scale_op_name, "bias_term", true);
  this->engine_->AddOpAttr(scale_op_name, "weight_1", *scale);
  this->engine_->AddOpAttr(scale_op_name, "weight_2", *bias);
}

}  // namespace anakin
}  // namespace inference
}  // namespace paddle

#ifdef PADDLE_WITH_CUDA
REGISTER_CUDA_ANAKIN_OP_CONVERTER(batch_norm,
                                  BatchNormOpConverter<::anakin::saber::NV>);
#endif

REGISTER_CPU_ANAKIN_OP_CONVERTER(batch_norm,
                                 BatchNormOpConverter<::anakin::saber::X86>);
