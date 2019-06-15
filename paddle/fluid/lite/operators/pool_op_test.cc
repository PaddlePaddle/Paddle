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

#include "paddle/fluid/lite/operators/pool_op.h"
#include <gtest/gtest.h>
#include "paddle/fluid/lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

TEST(pool_op_lite, test) {
  // prepare variables
  Scope scope;
  auto* x = scope.Var("x")->GetMutable<Tensor>();
  auto* output = scope.Var("output")->GetMutable<Tensor>();
  x->Resize(DDim(std::vector<int64_t>({1, 3, 224, 224})));
  output->Resize(DDim(std::vector<int64_t>{1, 3, 112, 112}));

  // set data
  for (int i = 0; i < 1 * 3 * 224 * 224; i++) {
    x->mutable_data<float>()[i] = i;
  }
  for (int i = 0; i < 1 * 3 * 112 * 112; i++) {
    output->mutable_data<float>()[i] = 0.;
  }

  // prepare op desc
  cpp::OpDesc desc;
  desc.SetType("pool2d");
  desc.SetInput("X", {"x"});
  desc.SetOutput("Out", {"output"});

  std::string pooling_type("max");
  desc.SetAttr("pooling_type", pooling_type);
  // desc.SetAttr("ksize", static_cast<std::vector<int>>({2, 2}));
  std::vector<int> ksize{2, 2};
  desc.SetAttr("ksize", ksize);

  bool global_pooling{false};
  desc.SetAttr("global_pooling", global_pooling);

  std::vector<int> strides{1, 1};
  desc.SetAttr("strides", strides);

  std::vector<int> paddings{0, 0};
  desc.SetAttr("paddings", paddings);

  bool exclusive{true};
  desc.SetAttr("exclusive", exclusive);

  bool adaptive{false};
  desc.SetAttr("adaptive", adaptive);

  bool ceil_mode{false};
  desc.SetAttr("ceil_mode", ceil_mode);

  bool use_quantizer{false};
  desc.SetAttr("use_quantizer", use_quantizer);

  PoolOpLite pool("pool2d");
  pool.SetValidPlaces({Place{TARGET(kARM), PRECISION(kFloat)}});
  pool.Attach(desc, &scope);
  auto kernels = pool.CreateKernels({Place{TARGET(kARM), PRECISION(kFloat)}});
  LOG(INFO) << "kernels.size(): " << kernels.size();
#ifdef LITE_WITH_ARM
  ASSERT_FALSE(kernels.empty());
#else
  ASSERT_TRUE(kernels.empty());
#endif
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

#ifdef LITE_WITH_ARM
USE_LITE_KERNEL(pool2d, kARM, kFloat, kNCHW, def);
#endif
