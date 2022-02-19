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
#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/core/infermeta_utils.h"
#include "paddle/pten/infermeta/generated.h"
#include "paddle/pten/infermeta/unary.h"

namespace pten {
namespace tests {

TEST(WrappedInferMeta, Scale) {
  pten::DenseTensor dense_x;
  dense_x.Resize(pten::framework::make_ddim({3, 4}));

  pten::MetaTensor meta_x(&dense_x);
  pten::DenseTensor dense_out1;
  pten::MetaTensor meta_out(&dense_out1);
  pten::ScaleInferMeta(meta_x, 0, 0, false, &meta_out);

  EXPECT_EQ(dense_out1.dims().size(), dense_x.dims().size());
  EXPECT_EQ(dense_out1.dims()[0], dense_x.dims()[0]);
  EXPECT_EQ(dense_out1.dims()[1], dense_x.dims()[1]);
}

TEST(MetaFnFactory, InferMetaFnExists) {
  pten::DenseTensor dense_x;
  dense_x.Resize(pten::framework::make_ddim({3, 4}));

  pten::MetaTensor meta_x(&dense_x);
  pten::DenseTensor dense_out1;
  pten::MetaTensor meta_out(&dense_out1);
  pten::UnchangedInferMeta(meta_x, &meta_out);

  auto shared_meat_x = std::make_shared<pten::MetaTensor>(&dense_x);
  pten::DenseTensor dense_out2;
  auto shared_meta_out = std::make_shared<pten::MetaTensor>(&dense_out2);
  pten::InferMetaContext ctx;
  ctx.EmplaceBackInput(shared_meat_x);
  ctx.EmplaceBackOutput(shared_meta_out);
  ctx.SetMetaConfig(/*is_runtime=*/true);
  pten::MetaFnFactory::Instance().Get("sign")(&ctx);

  EXPECT_EQ(dense_out1.dims().size(), dense_out2.dims().size());
  EXPECT_EQ(dense_out1.dims()[0], dense_out2.dims()[0]);
  EXPECT_EQ(dense_out1.dims()[1], dense_out2.dims()[1]);
}

TEST(MetaFnFactory, CopyInferMetaFn) {
  pten::DenseTensor dense_x;
  dense_x.Resize(pten::framework::make_ddim({3, 4}));

  pten::MetaTensor meta_x(&dense_x);
  pten::DenseTensor dense_out1;
  pten::MetaTensor meta_out(&dense_out1);
  pten::UnchangedInferMeta(meta_x, &meta_out);

  auto shared_meat_x = std::make_shared<pten::MetaTensor>(&dense_x);
  pten::DenseTensor dense_out2;
  auto shared_meta_out = std::make_shared<pten::MetaTensor>(&dense_out2);

  pten::InferMetaContext ctx;
  ctx.EmplaceBackInput(shared_meat_x);
  ctx.EmplaceBackAttr(Backend::CPU);
  ctx.EmplaceBackAttr(false);
  ctx.EmplaceBackOutput(shared_meta_out);
  ctx.SetMetaConfig(/*is_runtime=*/true);
  pten::MetaFnFactory::Instance().Get("copy_to")(&ctx);

  EXPECT_EQ(dense_out1.dims().size(), dense_out2.dims().size());
  EXPECT_EQ(dense_out1.dims()[0], dense_out2.dims()[0]);
  EXPECT_EQ(dense_out1.dims()[1], dense_out2.dims()[1]);
}

TEST(MetaFnFactory, SplitInferMetaFn) {
  pten::DenseTensor dense_x;
  dense_x.Resize(pten::framework::make_ddim({4, 10}));
  pten::MetaTensor meta_x(&dense_x);
  auto shared_meat_x = std::make_shared<pten::MetaTensor>(&dense_x);

  pten::DenseTensor dense_out1;
  pten::DenseTensor dense_out2;
  paddle::SmallVector<std::shared_ptr<pten::MetaTensor>> out;
  out.push_back(std::make_shared<pten::MetaTensor>(&dense_out1));
  out.push_back(std::make_shared<pten::MetaTensor>(&dense_out2));

  pten::InferMetaContext ctx;
  ctx.EmplaceBackInput(shared_meat_x);
  ScalarArray num_or_sections{2, 2};
  Scalar axis{0};
  ctx.EmplaceBackAttr(num_or_sections);
  ctx.EmplaceBackAttr(axis);
  ctx.EmplaceBackOutputs(out);
  ctx.SetMetaConfig(/*is_runtime=*/true);

  pten::MetaFnFactory::Instance().Get("split")(&ctx);

  ASSERT_EQ(dense_out1.dims().size(), 2);
  ASSERT_EQ(dense_out1.dims()[0], 2);
  ASSERT_EQ(dense_out1.dims()[1], 10);

  ASSERT_EQ(dense_out2.dims().size(), 2);
  ASSERT_EQ(dense_out2.dims()[0], 2);
  ASSERT_EQ(dense_out2.dims()[1], 10);
}

}  // namespace tests
}  // namespace pten
