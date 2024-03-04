// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/op_desc.h"
#include <complex>
#include "gtest/gtest.h"
#include "paddle/phi/common/scalar.h"

TEST(OpDesc, SetScalarAttr) {
  paddle::framework::OpDesc opdesc;
  paddle::experimental::Scalar scalar(std::complex<double>(42.1, 42.1));
  opdesc.SetPlainAttr("scalar", scalar);
  ASSERT_EQ(opdesc.GetAttrType("scalar"), paddle::framework::proto::SCALAR);
}

TEST(OpDesc, SetScalarsAttr) {
  paddle::framework::OpDesc opdesc;
  paddle::experimental::Scalar scalar(std::complex<double>(42.1, 42.1));

  std::vector<paddle::experimental::Scalar> scalars;
  scalars.reserve(4);
  for (int i = 0; i < 4; i++) {
    scalars.emplace_back(i);
  }
  opdesc.SetPlainAttr("scalars", scalars);
  ASSERT_EQ(opdesc.GetAttrType("scalars"), paddle::framework::proto::SCALARS);
  opdesc.Flush();
}
