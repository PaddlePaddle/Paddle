/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <iostream>

#include "gtest/gtest.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/unary.h"

namespace phi {
namespace tests {

TEST(MetaFnFactory, InferMetaFnExists) {
  phi::DenseTensor dense_x;
  dense_x.Resize(common::make_ddim({3, 4}));

  phi::MetaTensor meta_x(&dense_x);
  phi::DenseTensor dense_out1;
  phi::MetaTensor meta_out(&dense_out1);
  phi::UnchangedInferMeta(meta_x, &meta_out);
}

void TestEmptyVectorInputInferMeta(const std::vector<const MetaTensor*>& inputs,
                                   std::vector<MetaTensor*> outputs) {
  ASSERT_EQ(inputs.size(), 0UL);
  ASSERT_EQ(outputs.size(), 0UL);
}

TEST(MetaFnFactory, EmptyVectorInputInferMetaFn) {
  phi::InferMetaContext ctx;
  ctx.EmplaceBackInput(MetaTensor());
  ctx.EmplaceBackOutput(MetaTensor());

  PD_INFER_META(TestEmptyVectorInputInferMeta)(&ctx);
}

}  // namespace tests
}  // namespace phi
