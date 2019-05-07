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
#include "paddle/fluid/inference/anakin/convert/helper.h"

namespace paddle {
namespace inference {
namespace anakin {

template <typename TargetT, ::anakin::Precision PrecisionT>
void BatchNormOpConverter<TargetT, PrecisionT>::operator()(
    const framework::proto::OpDesc &op, const framework::BlockDesc &block_desc,
    const framework::Scope &scope, bool test_mode) {
  framework::OpDesc op_desc(op, nullptr);
  PADDLE_ENFORCE_EQ(op_desc.Output("Y").size(), 1);
  std::map<std::string, std::string> inputs;
  for (auto k : {"X", "Scale", "Bias", "Mean", "Variance"}) {
    PADDLE_ENFORCE_EQ(op_desc.Input(k).size(), 1UL);
  }

  auto input = op_desc.Input("X").front();
  auto output = op_desc.Output("Y").front();
  auto op_name = op_desc.Type() + ":" + op_desc.Output("Y").front();
  auto epsilon = boost::get<float>(op_desc.GetAttr("epsilon"));

  auto bn_op_name = op_name + ":bn";
  auto bn_output = bn_op_name + "_output";
  this->engine_->AddOp(bn_op_name, "BatchNorm", {input}, {bn_output});
  this->engine_->AddOpAttr(bn_op_name, "epsilon", epsilon);
  this->engine_->AddOpAttr(bn_op_name, "momentum", static_cast<float>(1.0));

  auto scale_op_name = op_name + ":scale";
  this->engine_->AddOp(scale_op_name, "Scale", {bn_output}, {output});
  this->engine_->AddOpAttr(scale_op_name, "axis", 1);
  this->engine_->AddOpAttr(scale_op_name, "num_axes", 1);
  this->engine_->AddOpAttr(scale_op_name, "bias_term", true);

  auto *mean_v = scope.FindVar(op_desc.Input("Mean").front());
  PADDLE_ENFORCE_NOT_NULL(mean_v);
  auto weight1 = pblock_from_var<TargetT, PrecisionT>(*mean_v, this->engine_);
  this->engine_->AddOpAttr(bn_op_name, "weight_1", *weight1);

  auto *variance_v = scope.FindVar(op_desc.Input("Variance").front());
  PADDLE_ENFORCE_NOT_NULL(variance_v);
  auto weight2 =
      pblock_from_var<TargetT, PrecisionT>(*variance_v, this->engine_);
  this->engine_->AddOpAttr(bn_op_name, "weight_2", *weight2);

  auto *weight3 = pblock_from_vector<TargetT, PrecisionT>(
      std::vector<float>({1}), this->engine_);
  this->engine_->AddOpAttr(bn_op_name, "weight_3", *weight3);

  auto *scale_v = scope.FindVar(op_desc.Input("Scale").front());
  PADDLE_ENFORCE_NOT_NULL(scale_v);
  auto scale = pblock_from_var<TargetT, PrecisionT>(*scale_v, this->engine_);
  this->engine_->AddOpAttr(scale_op_name, "weight_1", *scale);

  auto *bias_v = scope.FindVar(op_desc.Input("Bias").front());
  PADDLE_ENFORCE_NOT_NULL(bias_v);
  auto bias = pblock_from_var<TargetT, PrecisionT>(*bias_v, this->engine_);
  this->engine_->AddOpAttr(scale_op_name, "weight_2", *bias);
}

}  // namespace anakin
}  // namespace inference
}  // namespace paddle

REGISTER_ANAKIN_OP_CONVERTER(batch_norm, BatchNormOpConverter);
