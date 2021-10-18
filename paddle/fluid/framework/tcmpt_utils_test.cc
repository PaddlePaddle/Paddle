/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/tcmpt_utils.h"
#include "gtest/gtest.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/framework/variable.h"

namespace paddle {
namespace framework {

TEST(TcmptUtils, MakeTensor) {
  // 1. create tensor
  LoDTensor x;
  Tensor x2;
  x.Resize({2});
  x.mutable_data<float>(platform::CPUPlace());
  x.data<float>()[0] = 0.2;
  x.data<float>()[1] = 0.5;

  // 2. test API
  auto dense_x = MakeTensorImpl<pt::DenseTensor>(x, x.place(), x.type());

  // 3. check result
  std::vector<float> expect_value = {0.2, 0.5};
  ASSERT_EQ(dense_x->data<float>()[0], expect_value[0]);
  ASSERT_EQ(dense_x->data<float>()[1], expect_value[1]);
  ASSERT_EQ(dense_x->backend(), pt::Backend::kCPU);
  ASSERT_EQ(dense_x->type(), pt::DataType::kFLOAT32);
}

TEST(TcmptUtils, VarToPtTensor) {
  // 1. create Variable
  Variable v;
  auto selected_rows = v.GetMutable<SelectedRows>();
  Tensor* value = selected_rows->mutable_value();
  auto* data =
      value->mutable_data<int>(make_ddim({1, 1}), paddle::platform::CPUPlace());
  data[0] = 123;
  pt::Backend expect_backend = pt::Backend::kCPU;

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  expect_backend = pt::Backend::kCUDA;
#endif
  auto tensor_def = pt::TensorArgDef(expect_backend, pt::DataLayout::kNCHW,
                                     pt::DataType::kINT32);
  // 2. test API
  auto tensor_x = InputVariableToPtTensor(v, tensor_def);
  // 3. check result
  ASSERT_EQ(tensor_x->backend(), expect_backend);
  ASSERT_EQ(tensor_x->type(), pt::DataType::kINT32);
}

}  // namespace framework
}  // namespace paddle
