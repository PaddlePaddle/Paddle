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

#include "paddle/fluid/lite/operators/reshape_op.h"
#include <gtest/gtest.h>
#include <map>
#include <vector>
#include "paddle/fluid/lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

TEST(reshape_op_lite, test) {
  // prepare variables
  Scope scope;
  auto* x = scope.Var("x")->GetMutable<Tensor>();
  auto* actual_shape = scope.Var("actual_shape")->GetMutable<Tensor>();
  auto* output = scope.Var("output")->GetMutable<Tensor>();
  std::map<std::vector<int>, std::vector<int64_t>> shapes = {
      {{-1, 0, 3, 2, 1}, {2, 4, 3, 2, 1}},
      {{0, -1, 3, 2, 1}, {2, 4, 3, 2, 1}},
      {{-1, 48}, {1, 48}},
      {{48, -1}, {48, 1}},
      {{0, 24}, {2, 24}},
      {{12, 0}, {12, 4}},
  };
  x->Resize(DDim(std::vector<int64_t>({2, 4, 6})));
  actual_shape->Resize(DDim(std::vector<int64_t>({2})));

  auto* actual_shape_data = actual_shape->mutable_data<int>();
  actual_shape_data[0] = 6;
  actual_shape_data[1] = 8;

  for (auto& shape : shapes) {
    for (auto& has_actual_shape : {true, false}) {
      for (auto& inplace : {true, false}) {
        // prepare op desc
        cpp::OpDesc desc;
        desc.SetType("reshape");
        desc.SetInput("X", {"x"});
        if (has_actual_shape) {
          desc.SetInput("Shape", {"actual_shape"});
        }
        desc.SetOutput("Out", {"output"});
        desc.SetAttr("shape", shape.first);
        desc.SetAttr("inplace", inplace);

        ReshapeOp reshape("reshape");

        reshape.SetValidPlaces(
            {Place{TARGET(kHost), PRECISION(kAny), DATALAYOUT(kAny)}});
        reshape.Attach(desc, &scope);
        reshape.CheckShape();
        reshape.InferShape();

        // check output dims
        auto output_dims = output->dims();
        CHECK_EQ(output_dims.size(), shape.second.size());
        for (size_t i = 0; i < output_dims.size(); i++) {
          CHECK_EQ(output_dims[i], shape.second[i]);
        }
      }
    }
  }
}

TEST(reshape2_op_lite, test) {
  // prepare variables
  Scope scope;
  auto* x = scope.Var("x")->GetMutable<Tensor>();
  auto* actual_shape = scope.Var("actual_shape")->GetMutable<Tensor>();
  auto* output = scope.Var("output")->GetMutable<Tensor>();
  auto* xshape = scope.Var("xshape")->GetMutable<Tensor>();
  std::map<std::vector<int>, std::vector<int64_t>> shapes = {
      {{-1, 0, 3, 2, 1}, {2, 4, 3, 2, 1}},
      {{0, -1, 3, 2, 1}, {2, 4, 3, 2, 1}},
      {{-1, 48}, {1, 48}},
      {{48, -1}, {48, 1}},
      {{0, 24}, {2, 24}},
      {{12, 0}, {12, 4}},
  };
  x->Resize(DDim(std::vector<int64_t>({2, 4, 6})));
  actual_shape->Resize(DDim(std::vector<int64_t>({2})));

  auto* actual_shape_data = actual_shape->mutable_data<int>();
  actual_shape_data[0] = 6;
  actual_shape_data[1] = 8;

  for (auto& shape : shapes) {
    for (auto& has_actual_shape : {true, false}) {
      for (auto& inplace : {true, false}) {
        // prepare op desc
        cpp::OpDesc desc;
        desc.SetType("reshape");
        desc.SetInput("X", {"x"});
        if (has_actual_shape) {
          desc.SetInput("Shape", {"actual_shape"});
        }
        desc.SetOutput("Out", {"output"});
        desc.SetOutput("XShape", {"xshape"});
        desc.SetAttr("shape", shape.first);
        desc.SetAttr("inplace", inplace);

        Reshape2Op reshape2("reshape2");

        reshape2.SetValidPlaces(
            {Place{TARGET(kHost), PRECISION(kAny), DATALAYOUT(kAny)}});
        reshape2.Attach(desc, &scope);
        reshape2.CheckShape();
        reshape2.InferShape();

        // check output dims
        auto output_dims = output->dims();
        CHECK_EQ(output_dims.size(), shape.second.size());
        for (int i = 0; i < output_dims.size(); i++) {
          CHECK_EQ(output_dims[i], shape.second[i]);
        }
        // check xshape dims
        auto x_dims = x->dims();
        auto xshape_dims = xshape->dims();
        CHECK_EQ(xshape_dims.size(), x_dims.size() + 1);
        CHECK_EQ(xshape_dims[0], 0);
        for (size_t i = 0; i < x_dims.size(); i++) {
          CHECK_EQ(xshape_dims[i + 1], x_dims[i]);
        }
      }
    }
  }
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle
