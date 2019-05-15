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

#include "paddle/fluid/lite/core/type_system.h"
#include <gtest/gtest.h>

namespace paddle {
namespace lite {

TEST(TypeSystem, CheckDuplicateGet) {
  auto* tensor_ty =
      Type::GetTensorTy(TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW));
  auto* tensor_ty1 =
      Type::GetTensorTy(TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW));

  ASSERT_EQ(tensor_ty, tensor_ty1);

  ASSERT_EQ(tensor_ty->target(), TARGET(kHost));
  ASSERT_EQ(tensor_ty->precision(), PRECISION(kFloat));
  ASSERT_EQ(tensor_ty->layout(), DATALAYOUT(kNCHW));
}

}  // namespace lite
}  // namespace paddle
