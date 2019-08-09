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

#include "paddle/fluid/lite/operators/batch_norm_op.h"
#include <gtest/gtest.h>
#include "paddle/fluid/lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

TEST(batch_norm_op_lite, test) {
  // prepare variables
  Scope scope;
  auto* x = scope.Var("x")->GetMutable<Tensor>();
  auto* scale = scope.Var("scale")->GetMutable<Tensor>();
  auto* bias = scope.Var("bias")->GetMutable<Tensor>();
  auto* mean = scope.Var("mean")->GetMutable<Tensor>();
  auto* variance = scope.Var("variance")->GetMutable<Tensor>();
  auto* y = scope.Var("y")->GetMutable<Tensor>();
  x->Resize({2, 32, 10, 20});
  auto x_dims = x->dims();
  const int64_t channel_size = x_dims[1];  // NCHW
  scale->Resize({channel_size});
  bias->Resize({channel_size});
  mean->Resize({channel_size});
  variance->Resize({channel_size});

  // prepare op desc
  cpp::OpDesc desc;
  desc.SetType("batch_norm");
  desc.SetInput("X", {"x"});
  desc.SetInput("Scale", {"scale"});
  desc.SetInput("Bias", {"bias"});
  desc.SetInput("Mean", {"mean"});
  desc.SetInput("Variance", {"variance"});
  desc.SetOutput("Y", {"y"});
  desc.SetAttr("is_test", static_cast<int>(1));
  desc.SetAttr("use_global_stats", false);
  desc.SetAttr("epsilon", 1e-5f);
  desc.SetAttr("momentum", 0.9f);
  desc.SetAttr("data_layout", std::string("NCHW"));

  BatchNormOp batch_norm("batch_norm");

  batch_norm.SetValidPlaces({Place{TARGET(kHost), PRECISION(kFloat)}});
  batch_norm.Attach(desc, &scope);
  batch_norm.CheckShape();
  batch_norm.InferShape();

  // check output dims
  auto y_dims = y->dims();
  CHECK_EQ(y_dims.size(), x_dims.size());
  for (size_t i = 0; i < y_dims.size(); i++) {
    CHECK_EQ(y_dims[i], x_dims[i]);
  }
}

TEST(batch_norm_op_lite, test_enable_is_test) {
  // prepare variables
  Scope scope;
  auto* x = scope.Var("x")->GetMutable<Tensor>();
  auto* scale = scope.Var("scale")->GetMutable<Tensor>();
  auto* bias = scope.Var("bias")->GetMutable<Tensor>();
  auto* mean = scope.Var("mean")->GetMutable<Tensor>();
  auto* variance = scope.Var("variance")->GetMutable<Tensor>();
  auto* y = scope.Var("y")->GetMutable<Tensor>();
  auto* mean_out = scope.Var("mean_out")->GetMutable<Tensor>();
  auto* variance_out = scope.Var("variance_out")->GetMutable<Tensor>();
  auto* saved_mean = scope.Var("saved_mean")->GetMutable<Tensor>();
  auto* saved_variance = scope.Var("saved_variance")->GetMutable<Tensor>();
  x->Resize({2, 32, 10, 20});
  auto x_dims = x->dims();
  const int64_t channel_size = x_dims[1];  // NCHW
  scale->Resize({channel_size});
  bias->Resize({channel_size});
  mean->Resize({channel_size});
  variance->Resize({channel_size});

  // prepare op desc
  cpp::OpDesc desc;
  desc.SetType("batch_norm");
  desc.SetInput("X", {"x"});
  desc.SetInput("Scale", {"scale"});
  desc.SetInput("Bias", {"bias"});
  desc.SetInput("Mean", {"mean"});
  desc.SetInput("Variance", {"variance"});
  desc.SetOutput("Y", {"y"});
  desc.SetOutput("MeanOut", {"mean_out"});
  desc.SetOutput("VarianceOut", {"variance_out"});
  desc.SetOutput("SavedMean", {"saved_mean"});
  desc.SetOutput("SavedVariance", {"saved_variance"});
  desc.SetAttr("is_test", static_cast<int>(0));
  desc.SetAttr("use_global_stats", false);
  desc.SetAttr("epsilon", 1e-5f);
  desc.SetAttr("momentum", 0.9f);
  desc.SetAttr("data_layout", std::string("NCHW"));

  BatchNormOp batch_norm("batch_norm");

  batch_norm.SetValidPlaces({Place{TARGET(kHost), PRECISION(kFloat)}});
  batch_norm.Attach(desc, &scope);
  batch_norm.CheckShape();
  batch_norm.InferShape();

  // check output dims
  auto y_dims = y->dims();
  CHECK_EQ(y_dims.size(), x_dims.size());
  for (size_t i = 0; i < y_dims.size(); i++) {
    CHECK_EQ(y_dims[i], x_dims[i]);
  }
  auto mean_out_dims = mean_out->dims();
  auto variance_out_dims = variance_out->dims();
  auto saved_mean_dims = saved_mean->dims();
  auto saved_variance_dims = saved_variance->dims();
  CHECK_EQ(mean_out_dims.size(), 1UL);
  CHECK_EQ(variance_out_dims.size(), 1UL);
  CHECK_EQ(saved_mean_dims.size(), 1UL);
  CHECK_EQ(saved_variance_dims.size(), 1UL);
  CHECK_EQ(mean_out_dims[0], channel_size);
  CHECK_EQ(variance_out_dims[0], channel_size);
  CHECK_EQ(saved_mean_dims[0], channel_size);
  CHECK_EQ(saved_variance_dims[0], channel_size);
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle
