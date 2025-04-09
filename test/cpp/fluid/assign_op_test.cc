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

#include "paddle/common/ddim.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/phi/common/place.h"

TEST(AssignOp, AssignDenseTensor) {
  phi::CPUPlace cpu_place;
  phi::CPUContext ctx(cpu_place);

  paddle::framework::Variable output;
  paddle::operators::AssignFunctor assign_functor(&output, ctx);

  phi::DenseTensor input;
  phi::DDim in_dims = common::make_ddim({3, 4});
  int* in_data = input.mutable_data<int>(in_dims, cpu_place);
  for (int i = 0; i < 12; ++i) {
    in_data[i] = i;
  }

  assign_functor(input);

  auto& out_tensor = output.Get<phi::DenseTensor>();
  phi::DDim out_dims = out_tensor.dims();
  EXPECT_EQ(in_dims, out_dims);
  auto* out_data = out_tensor.data<int>();
  for (int i = 0; i < 12; ++i) {
    EXPECT_EQ(i, out_data[i]);
  }
}

TEST(AssignOp, AssignDenseTensorArray) {
  phi::CPUPlace cpu_place;
  phi::CPUContext ctx(cpu_place);

  paddle::framework::Variable output;
  paddle::operators::AssignFunctor assign_functor(&output, ctx);

  phi::TensorArray input;
  for (int i = 0; i < 5; ++i) {
    phi::DDim in_dims = common::make_ddim({i + 1, i + 2});
    phi::DenseTensor dense_tensor;
    float* in_data = dense_tensor.mutable_data<float>(in_dims, cpu_place);
    for (int j = 0; j < (i + 1) * (i + 2); ++j) {
      in_data[j] = static_cast<float>(j);
    }
    input.push_back(dense_tensor);
  }

  assign_functor(input);

  auto& out_array = output.Get<phi::TensorArray>();
  for (int i = 0; i < 5; ++i) {
    phi::DDim out_dims = out_array[i].dims();
    EXPECT_EQ(common::make_ddim({i + 1, i + 2}), out_dims);
    const float* out_data = out_array[i].data<float>();
    for (int j = 0; j < (i + 1) * (i + 2); ++j) {
      EXPECT_EQ(static_cast<float>(j), out_data[j]);
    }
  }
}

TEST(AssignOp, AssignSelectedRows) {
  phi::CPUPlace cpu_place;
  phi::CPUContext ctx(cpu_place);

  paddle::framework::Variable output;
  paddle::operators::AssignFunctor assign_functor(&output, ctx);

  std::vector<int64_t> rows{0, 4, 7};
  int64_t height = 10;

  phi::SelectedRows input(rows, height);
  phi::DenseTensor* input_tensor = input.mutable_value();

  phi::DDim in_dims = common::make_ddim({3, 4});
  int* in_data = input_tensor->mutable_data<int>(in_dims, cpu_place);
  for (int i = 0; i < 12; ++i) {
    in_data[i] = i;
  }

  assign_functor(input);

  auto& out_selected_row = output.Get<phi::SelectedRows>();
  const phi::Vector<int64_t>& out_rows = out_selected_row.rows();
  EXPECT_EQ(rows.size(), out_rows.size());
  for (size_t i = 0; i < rows.size(); ++i) {
    EXPECT_EQ(rows[i], out_rows[i]);
  }
  EXPECT_EQ(height, out_selected_row.height());
  const phi::DenseTensor& out_tensor = out_selected_row.value();
  phi::DDim out_dims = out_tensor.dims();
  EXPECT_EQ(in_dims, out_dims);
  auto* out_data = out_tensor.data<int>();
  for (int i = 0; i < 12; ++i) {
    EXPECT_EQ(i, out_data[i]);
  }
}
