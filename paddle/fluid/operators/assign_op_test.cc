/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#include "paddle/fluid/operators/assign_op.h"

#include <gtest/gtest.h>

#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/platform/place.h"

TEST(AssignOp, AssignLoDTensor) {
  paddle::platform::CPUPlace cpu_place;
  paddle::platform::CPUDeviceContext ctx(cpu_place);

  paddle::framework::Variable output;
  paddle::operators::AssignFunctor assign_functor(&output, ctx);

  paddle::framework::LoDTensor input;
  paddle::framework::DDim in_dims = paddle::framework::make_ddim({3, 4});
  int* in_data = input.mutable_data<int>(in_dims, cpu_place);
  for (int i = 0; i < 12; ++i) {
    in_data[i] = i;
  }

  assign_functor(input);

  auto& out_tensor = output.Get<paddle::framework::LoDTensor>();
  paddle::framework::DDim out_dims = out_tensor.dims();
  EXPECT_EQ(in_dims, out_dims);
  auto* out_data = out_tensor.data<int>();
  for (int i = 0; i < 12; ++i) {
    EXPECT_EQ(i, out_data[i]);
  }
}

TEST(AssignOp, AssignLoDTensorArray) {
  paddle::platform::CPUPlace cpu_place;
  paddle::platform::CPUDeviceContext ctx(cpu_place);

  paddle::framework::Variable output;
  paddle::operators::AssignFunctor assign_functor(&output, ctx);

  paddle::framework::LoDTensorArray input;
  for (int i = 0; i < 5; ++i) {
    paddle::framework::DDim in_dims =
        paddle::framework::make_ddim({i + 1, i + 2});
    paddle::framework::LoDTensor lod_tensor;
    float* in_data = lod_tensor.mutable_data<float>(in_dims, cpu_place);
    for (int j = 0; j < (i + 1) * (i + 2); ++j) {
      in_data[j] = static_cast<float>(j);
    }
    input.push_back(lod_tensor);
  }

  assign_functor(input);

  auto& out_array = output.Get<paddle::framework::LoDTensorArray>();
  for (int i = 0; i < 5; ++i) {
    paddle::framework::DDim out_dims = out_array[i].dims();
    EXPECT_EQ(paddle::framework::make_ddim({i + 1, i + 2}), out_dims);
    const float* out_data = out_array[i].data<float>();
    for (int j = 0; j < (i + 1) * (i + 2); ++j) {
      EXPECT_EQ(static_cast<float>(j), out_data[j]);
    }
  }
}

TEST(AssignOp, AssignSelectedRows) {
  paddle::platform::CPUPlace cpu_place;
  paddle::platform::CPUDeviceContext ctx(cpu_place);

  paddle::framework::Variable output;
  paddle::operators::AssignFunctor assign_functor(&output, ctx);

  std::vector<int64_t> rows{0, 4, 7};
  int64_t height = 10;

  pten::SelectedRows input(rows, height);
  paddle::framework::Tensor* input_tensor = input.mutable_value();

  paddle::framework::DDim in_dims = paddle::framework::make_ddim({3, 4});
  int* in_data = input_tensor->mutable_data<int>(in_dims, cpu_place);
  for (int i = 0; i < 12; ++i) {
    in_data[i] = i;
  }

  assign_functor(input);

  auto& out_selected_row = output.Get<pten::SelectedRows>();
  const paddle::framework::Vector<int64_t>& out_rows = out_selected_row.rows();
  EXPECT_EQ(rows.size(), out_rows.size());
  for (size_t i = 0; i < rows.size(); ++i) {
    EXPECT_EQ(rows[i], out_rows[i]);
  }
  EXPECT_EQ(height, out_selected_row.height());
  const paddle::framework::Tensor& out_tensor = out_selected_row.value();
  paddle::framework::DDim out_dims = out_tensor.dims();
  EXPECT_EQ(in_dims, out_dims);
  auto* out_data = out_tensor.data<int>();
  for (int i = 0; i < 12; ++i) {
    EXPECT_EQ(i, out_data[i]);
  }
}
